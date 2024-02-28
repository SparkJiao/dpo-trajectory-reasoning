import datetime
import logging
import os
import sys

import deepspeed
import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import distributed as dist
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (PreTrainedTokenizer)

from data.mini_dataset import MiniDataset
from general_util.logger import setting_logger
from general_util.training_utils import (
    batch_to_device,
    set_seed,
    note_best_checkpoint,
    load_and_cache_examples,
    organize_multiple_dataset,
)
import fairscale.nn.model_parallel.initialize as mpu
from general_util.dist_utils import get_pipeline_parallel_world_size, get_pipeline_parallel_rank, prepare_distributed_sampler

logger: logging.Logger

"""
This script is used for PPO and ReMax training.
Check the example config `conf/exp/remax/v1_6.yaml`.
"""


def evaluation_by_reward(
        cfg: DictConfig, trainer, tokenizer, print_answers=False, prefix="", _split="dev",
):
    dataset = load_and_cache_examples(cfg, tokenizer, _split=_split)

    output_dir = getattr(cfg, "predict_dir", cfg.output_dir)

    if cfg.local_rank in [-1, 0] and not os.path.exists(os.path.join(output_dir, prefix)):
        os.makedirs(os.path.join(output_dir, prefix))

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    if cfg.ddp_eval and cfg.local_rank != -1:
        eval_sampler = DistributedSampler(dataset, shuffle=False)
    else:
        eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly

    eval_collator_cfg = getattr(cfg, f"{_split}_collator", None)
    if eval_collator_cfg is not None:
        eval_collator = hydra.utils.instantiate(eval_collator_cfg)
    else:
        eval_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=cfg.eval_batch_size,
                                 collate_fn=eval_collator,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True,
                                 prefetch_factor=cfg.prefetch_factor)

    # post_processor = hydra.utils.instantiate(cfg.post_process) if "post_process" in cfg and cfg.post_process else None

    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)

    trainer.eval()
    # eval_forward_fn = hydra.utils.instantiate(cfg.eval_forward_fn, cfg, model, tokenizer)

    for step, batch_prompt in enumerate(eval_dataloader):
        batch_prompt = batch_to_device(batch_prompt, cfg.device)

        exp = trainer.generate_experience(
            **batch_prompt,
            global_step=step,
            print_answers=True,
            training_mode=False,
        )
        # reward = exp["rewards"].mean()
        #
        # prompt_length = trainer.prompt_length
        # start = prompt_length - 1
        # action_mask = exp["attention_mask"]
        # answer_length = action_mask[:, start:].sum(dim=-1).float().mean()
        #
        # if "full_kl" in exp:
        #     kl = (
        #             torch.sum(exp["full_kl"][:, start:] * action_mask[:, start:])
        #             / action_mask[:, start:].sum()
        #     )
        # else:
        #     kl = (
        #             torch.sum(
        #                 (exp["logprobs"][:, start:] - exp["ref_logprobs"][:, start:])
        #                 * action_mask[:, start:-1]
        #             )
        #             / action_mask[:, start:-1].sum()
        #     )
        # if "entropy" in exp:
        #     entropy = (
        #             torch.sum(exp["entropy"][:, start:] * action_mask[:, start:])
        #             / action_mask[:, start:].sum()
        #     )
        # else:
        #     entropy = torch.zeros(1)
        #
        # eval_reward.append(reward.item())
        # eval_length.append(answer_length.item())
        # eval_kl.append(kl.item())
        # eval_entropy.append(entropy.item())

        # save eval result
        # if args.save_answers and step < 10:
        #     assert global_step is not None and args.output_dir is not None
        #     save_dir = os.path.join(args.output_dir, "evaluation")
        #     os.makedirs(save_dir, exist_ok=True)
        #
        #     prompts = trainer.tokenizer.batch_decode(
        #         exp["input_ids"][:, :prompt_length], skip_special_tokens=True
        #     )
        #     answers = trainer.tokenizer.batch_decode(
        #         exp["input_ids"][:, prompt_length:], skip_special_tokens=True
        #     )
        #     rewards = [rew.item() for rew in exp["rewards"]]
        #
        #     file_path = os.path.join(save_dir, f"rank_{args.local_rank}.json")
        #     save_prompts_and_answers(prompts, answers, rewards, global_step, file_path)
        #
        # if step == 19:
        #     break

        trainer.post_process_experience(exp)

    metrics = trainer.get_eval_metrics()
    return metrics


@hydra.main(config_path="conf", config_name="test", version_base="1.2")
def main(cfg: DictConfig):
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
        cfg.dp_size = 1
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=7200))
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
        cfg.dp_size = dist.get_world_size()
        if cfg.tp_size > 1:
            mpu.initialize_model_parallel(cfg.tp_size)
            cfg.dp_size = mpu.get_data_parallel_world_size()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")

    if mpu.model_parallel_is_initialized():
        dp_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        mp_size = mpu.get_model_parallel_world_size()
        mp_rank = mpu.get_model_parallel_rank()
        pp_size = get_pipeline_parallel_world_size()
        pp_rank = get_pipeline_parallel_rank()
        logger.warning(f"Rank: {cfg.local_rank}, "
                       f"Data Parallel: {dp_rank}/{dp_size}, "
                       f"Model Parallel: {mp_rank}/{mp_size}, "
                       f"Pipeline Parallel: {pp_rank}/{pp_size}")

    # Set seed
    set_seed(cfg)

    tokenizer: PreTrainedTokenizer = instantiate(cfg.tokenizer_init)

    # logger.info("Training/evaluation parameters %s", OmegaConf.to_yaml(cfg))
    if cfg.local_rank in [-1, 0] and cfg.do_train:
        if not os.path.exists(cfg.output_dir):
            os.makedirs(cfg.output_dir)
        OmegaConf.save(cfg, os.path.join(cfg.output_dir, "training_config.yaml"))

    cfg.train_batch_size = cfg.per_gpu_train_batch_size

    train_files, total_dataset_len = organize_multiple_dataset(cfg, tokenizer, _split="train")

    dp_degree = cfg.dp_size
    _actual_train_batch_size = cfg.train_batch_size * cfg.gradient_accumulation_steps * dp_degree
    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (cfg.total_dataset_len * cfg.generation_batches * cfg.ppo_epochs // _actual_train_batch_size) + 1
    else:
        t_total = cfg.total_dataset_len // _actual_train_batch_size * cfg.generation_batches * cfg.num_train_epochs * cfg.ppo_epochs
        cfg.max_steps = t_total

    if cfg.warmup_proportion:
        num_warmup_steps = int(t_total * cfg.warmup_proportion)
        cfg.warmup_steps = num_warmup_steps
    else:
        num_warmup_steps = cfg.warmup_steps

    rl_engine = instantiate(cfg.rl_engine_init, cfg)
    rl_trainer = instantiate(cfg.rl_trainer_init, rl_engine, cfg)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(cfg.generation_batches, cfg.per_gpu_train_batch_size)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", cfg.total_dataset_len)
    logger.info("  Num Epochs = %d", cfg.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", cfg.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", _actual_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", cfg.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup steps = %d", num_warmup_steps)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(int(cfg.num_train_epochs), desc="Epoch", disable=cfg.local_rank not in [-1, 0])
    set_seed(cfg)  # Added here for reproducibility (even between python 2 and 3)

    if cfg.resume:
        continue_from_global_step = int(cfg.resume.split('-')[-1])
        logger.info("Fast forwarding to global step %d to resume training from latest checkpoint...", continue_from_global_step)
        rl_trainer.resume(cfg.resume)
    else:
        continue_from_global_step = 0

    if cfg.local_rank in [-1, 0]:
        wandb.init(
            project="llm-reasoning",
            name=cfg.exp_name,
            notes=cfg.exp_notes,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.define_metric(cfg.prediction_cfg.metric, summary=("max" if cfg.prediction_cfg.measure > 0 else "min"))

        tb_helper = instantiate(cfg.summary_helper) if "summary_helper" in cfg and cfg.summary_helper else None
    else:
        tb_helper = None
    per_device_generation_batch_size = getattr(cfg, "per_device_generation_batch_size", cfg.per_gpu_train_batch_size)
    for epoch in train_iterator:
        for _file in train_files:
            sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
            if cfg.local_rank == -1:
                sub_train_sampler = RandomSampler(sub_train_dataset)
            else:
                sub_train_sampler = prepare_distributed_sampler(sub_train_dataset, cfg.seed)
            sub_train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
            sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                              sampler=sub_train_sampler,
                                              batch_size=per_device_generation_batch_size,
                                              collate_fn=sub_train_collator,
                                              num_workers=cfg.num_workers,
                                              pin_memory=True,
                                              prefetch_factor=cfg.prefetch_factor)

            epoch_iterator = tqdm(sub_train_dataloader, desc="Iteration", disable=cfg.local_rank not in [-1, 0], dynamic_ncols=True)
            if cfg.local_rank != -1:
                sub_train_dataloader.sampler.set_epoch(epoch)

            if dist.is_initialized():
                dist.barrier()

            step = 0
            for _, batch in enumerate(epoch_iterator):
                if global_step < continue_from_global_step:
                    for _ in range(cfg.generations_batches):  # TODO: Here is problem.
                        for _ in range((per_device_generation_batch_size // cfg.per_gpu_train_batch_size) * cfg.ppo_epochs):
                            step += 1
                            if step % cfg.gradient_accumulation_steps == 0:
                                global_step += 1
                    continue

                batch = batch_to_device(batch, device)

                exp_dataset = None
                for _ in range(cfg.generation_batches):
                    out = rl_trainer.generate_experience(
                        **batch,
                        global_step=global_step,
                    )

                    exp_dataset = exp_mini_dataset.add(out)

                assert exp_dataset is not None
                for ppo_ep in range(cfg.ppo_epochs):
                    for _, exp_data in enumerate(exp_dataset):
                        step += 1
                        rl_trainer.train()
                        outputs = rl_trainer.train_rl_step(exp_data)

                        if cfg.local_rank in [-1, 0]:
                            tb_helper.update(last_batch=None, last_outputs=outputs)

                        log_metrics = {}
                        if step % cfg.gradient_accumulation_steps == 0:
                            global_step += 1
                            if cfg.local_rank in [-1, 0]:
                                logs = tb_helper(clear=True)
                                log_metrics.update({f"train/{k}": v for k, v in logs.items()})
                                actor_lr = rl_engine.actor_lr_scheduler.get_lr()[0]
                                log_metrics["actor_lr"] = actor_lr

                            # Save model checkpoint
                            if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                                rl_trainer.save_model(cfg.output_dir, global_step=global_step)
                                logger.info("Saving model checkpoint to %s", cfg.output_dir)

                            # Evaluation
                            if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                                if cfg.ddp_eval or cfg.local_rank in [-1, 0]:
                                    results = evaluation_by_reward(cfg, rl_trainer, tokenizer, prefix=str(global_step), _split="dev")

                                    if cfg.local_rank in [-1, 0]:
                                        for key, value in results.items():
                                            log_metrics[f"eval/{key}"] = value

                                    sub_path = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                                    flag = note_best_checkpoint(cfg, results, sub_path)
                                    if cfg.save_best and flag:
                                        rl_trainer.save_model(cfg.output_dir, global_step=-1)

                            if len(log_metrics) > 0 and cfg.local_rank in [-1, 0]:
                                wandb.log(log_metrics)

                # if args.actor_gradient_checkpointing:
                #     rlhf_engine.actor.gradient_checkpointing_disable()
                #
                # actor_overflow = trainer.get_overflow()
                #
                # if not actor_overflow:
                #     non_overflow_step_count += 1

        #         if eval_kl >= args.target_kl:
        #             print_rank_0(
        #                 f"**** Early stop at {global_step} due to KL = {eval_kl} > Target KL ({args.target_kl}) ****"
        #             )
        #             early_stopped = True
        #             break
        #
        #         if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
        #             break
        #
        #     if args.enable_test_mode:
        #         break
        #
        # # Final
        # if not early_stopped or args.save_at_final:
        #     print_rank_0(f"***** Evaluating at final *****", args.global_rank)
        #     perplexity = evaluation_by_ppl(trainer, ppl_eval_dataloader, device)
        #     eval_reward, eval_length, eval_kl, eval_entropy = evaluation_by_reward(
        #         trainer, prompt_eval_dataloader, device, args, global_step, False
        #     )
        #     print_rank_0(
        #         f"eval reward: {eval_reward} | eval length: {eval_length} | eval kl: {eval_kl} | eval entropy: {eval_entropy} | eval ppl: {perplexity}",
        #         args.global_rank,
        #     )
        #     if args.enable_tensorboard and torch.distributed.get_rank() == 0:
        #         writer.add_scalar("eval/reward", eval_reward, global_step=global_step)
        #         writer.add_scalar("eval/length", eval_length, global_step=global_step)
        #         writer.add_scalar("eval/kl", eval_kl, global_step=global_step)
        #         writer.add_scalar("eval/entropy", eval_entropy, global_step=global_step)
        #         writer.add_scalar("eval/ppl", perplexity, global_step=global_step)
        #         writer.flush()
        #     if eval_reward >= best_eval_reward or args.save_at_final:
        #         best_eval_reward = eval_reward
        #         save_model(rlhf_engine, tokenizer, args)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    print("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    print("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()
