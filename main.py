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
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=7200))
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")

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

    dp_degree = dist.get_world_size() if cfg.local_rank != -1 else 1
    _actual_train_batch_size = cfg.train_batch_size * cfg.gradient_accumulation_steps * dp_degree
    if cfg.max_steps > 0:
        t_total = cfg.max_steps
        cfg.num_train_epochs = cfg.max_steps // (cfg.total_dataset_len * cfg.generation_batches // _actual_train_batch_size) + 1
    else:
        t_total = cfg.total_dataset_len // _actual_train_batch_size * cfg.generation_batches * cfg.num_train_epochs
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
    # unsup_mini_dataset = MiniDataset(cfg.generation_batches, cfg.per_device_training_batch_size)

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

    for epoch in train_iterator:
        for _file in train_files:
            sub_train_dataset = load_and_cache_examples(cfg, tokenizer, _split="train", _file=_file)
            sub_train_sampler = RandomSampler(sub_train_dataset) if cfg.local_rank == -1 else DistributedSampler(sub_train_dataset)
            sub_train_collator = hydra.utils.instantiate(cfg.collator) if "collator" in cfg and cfg.collator else None
            sub_train_dataloader = DataLoader(dataset=sub_train_dataset,
                                              sampler=sub_train_sampler,
                                              batch_size=cfg.train_batch_size,
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
                    for _ in range(cfg.generations_batches):
                        step += 1
                        if step % cfg.gradient_accumulation_steps == 0:
                            global_step += 1
                    continue

                batch = batch_to_device(batch, device)

                exp_dataset = None
                for _ in range(cfg.generation_batches):
                    out = rl_trainer.generate_experience(
                        # batch["input_ids"],
                        # batch["attention_mask"],
                        # batch["labels"],
                        **batch,
                        global_step=global_step,
                        # print_answers=cfg.print_answers and global_step % 20 == 0,
                    )

                    # if batch_unsupervised is not None:
                    #     batch_unsupervised = to_device(batch_unsupervised, device)
                    #     unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
                    # else:
                    # unsup_dataset = unsup_mini_dataset.add(
                    #     [[None] * args.per_device_generation_batch_size]
                    # )

                    exp_dataset = exp_mini_dataset.add(out)

                # training_start = time.time()
                assert exp_dataset is not None

                # inner_iter = 0
                # actor_loss_sum, unsup_loss_sum = 0, 0
                # average_reward = 0
                # average_return = 0
                # average_max_return = 0
                # average_length = 0
                # average_kl = 0
                # average_max_kl = 0
                # average_entropy = 0
                # average_kl_ratio = 0

                # if cfg.actor_gradient_checkpointing:
                #     rl_engine.actor.gradient_checkpointing_enable()

                # we manully implement gradient accumulation here
                # for i, (exp_data, unsup_data) in enumerate(
                #     zip(exp_dataset, unsup_dataset)
                # ):
                for _, exp_data in enumerate(exp_dataset):
                    step += 1
                    rl_trainer.train()
                    # actor_loss, actor_return, kl_ratio = rl_trainer.compute_loss(exp_data)
                    outputs = rl_trainer.train_rl_step(exp_data)

                    # rl_trainer.actor_model.backward(actor_loss / len(exp_dataset))

                    # actor_loss_sum += actor_loss.item()
                    # average_reward += exp_data["rewards"].mean()
                    # average_return += actor_return.mean()
                    # average_max_return += torch.abs(actor_return).max()
                    # average_kl_ratio += kl_ratio

                    # prompt_length = rl_trainer.prompt_length
                    # start = prompt_length - 1
                    # action_mask = exp_data["attention_mask"]
                    # answer_length = action_mask[:, start:].sum(dim=-1).float().mean()
                    # average_length += answer_length

                    # if "full_kl" in exp_data:
                    #     kl = (
                    #         torch.sum(
                    #             exp_data["full_kl"][:, start:] * action_mask[:, start:]
                    #         )
                    #         / action_mask[:, start:].sum()
                    #     )
                    #     max_kl = torch.max(
                    #         exp_data["full_kl"][:, start:] * action_mask[:, start:]
                    #     )
                    # else:
                    #     kl = (
                    #         torch.sum(
                    #             (
                    #                 exp_data["logprobs"][:, start:]
                    #                 - exp_data["ref_logprobs"][:, start:]
                    #             )
                    #             * action_mask[:, start:-1]
                    #         )
                    #         / action_mask[:, start:-1].sum()
                    #     )
                    #     max_kl = torch.max(
                    #         (
                    #             exp_data["logprobs"][:, start:]
                    #             - exp_data["ref_logprobs"][:, start:]
                    #         )
                    #         * action_mask[:, start:-1]
                    #     )
                    # if "entropy" in exp_data:
                    #     entropy = (
                    #         torch.sum(
                    #             exp_data["entropy"][:, start:] * action_mask[:, start:]
                    #         )
                    #         / action_mask[:, start:].sum()
                    #     )
                    # else:
                    #     entropy = torch.zeros(1)
                    # average_kl += kl
                    # average_entropy += entropy
                    # average_max_kl += max_kl

                    # if unsupervised_training_enabled:
                    #     raise NotImplementedError
                    #     # unsup_loss = trainer.train_unsupervised(
                    #     #     unsup_data, args.unsup_coef)
                    #     # unsup_loss_sum += unsup_loss.item()

                    # inner_iter += 1
                    # if args.enable_ema:
                    #     raise NotImplementedError

                    if cfg.local_rank in [-1, 0]:
                        tb_helper.update(last_batch=None, last_outputs=outputs)

                # ==== perform batch_update here (i.e., gradient checkpointing for on-policy)
                # rl_trainer.actor_model.step()
                # ====

                # end = time.time()
                # training_time = end - training_start
                # e2e_time = (
                #     training_time + rl_trainer.generate_time * cfg.generation_batches
                # )  # it is an approximation, we did not include, e.g., rw forward time etc

                # print_rank_0(
                #     f"Epoch: {epoch + 1}/{args.num_train_epochs} | Step: {step}/{len(prompt_train_dataloader)} |  Actor Loss: {actor_loss_sum / inner_iter} | Unsupervised Loss: {unsup_loss_sum / inner_iter}",
                #     args.global_rank,
                # )
                # print_throughput_step3(
                #     rlhf_engine.actor.module,
                #     args,
                #     e2e_time,
                #     trainer.generate_time,
                #     training_time,
                #     args.global_rank,
                # )
                # average_reward = get_all_reduce_mean(average_reward).item()
                # average_return = get_all_reduce_mean(average_return).item()
                # average_length = get_all_reduce_mean(average_length).item()
                # average_kl = get_all_reduce_mean(average_kl).item()
                # average_entropy = get_all_reduce_mean(average_entropy).item()
                # average_max_kl = get_all_reduce_mean(average_max_kl).item()
                # average_max_return = get_all_reduce_mean(average_max_return).item()
                # average_kl_ratio = get_all_reduce_mean(average_kl_ratio).item()
                # print_rank_0(
                #     f"Average reward score: {average_reward / inner_iter} Average return: {average_return / inner_iter} Average length: {average_length / inner_iter:.0f} Average kl: {average_kl / inner_iter} Average entropy: {average_entropy / inner_iter} Max kl: {average_max_kl / inner_iter} Max return: {average_max_return / inner_iter} KL ratio: {average_kl_ratio / inner_iter}",
                #     args.global_rank,
                # )
                # print_rank_0(
                #     "-------------------------------------------------------------------------------------",
                #     args.global_rank,
                # )

                # if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                #     writer.add_scalar(
                #         "train/reward",
                #         average_reward / inner_iter,
                #         global_step=global_step,
                #     )
                #     writer.add_scalar(
                #         "train/return",
                #         average_return / inner_iter,
                #         global_step=global_step,
                #     )
                #     writer.add_scalar(
                #         "train/length",
                #         average_length / inner_iter,
                #         global_step=global_step,
                #     )
                #     writer.add_scalar(
                #         "train/kl", average_kl / inner_iter, global_step=global_step
                #     )
                #     writer.add_scalar(
                #         "train/entropy",
                #         average_entropy / inner_iter,
                #         global_step=global_step,
                #     )
                #     writer.add_scalar(
                #         "train/actor_loss", actor_loss, global_step=global_step
                #     )
                #     writer.add_scalar(
                #         "train/actor_loss_sum", actor_loss_sum, global_step=global_step
                #     )
                #     writer.add_scalar(
                #         "train/max_kl",
                #         average_max_kl / inner_iter,
                #         global_step=global_step,
                #     )
                #     writer.add_scalar(
                #         "train/max_return",
                #         average_max_return / inner_iter,
                #         global_step=global_step,
                #     )
                #     writer.add_scalar(
                #         "train/kl_ratio",
                #         average_kl_ratio / inner_iter,
                #         global_step=global_step,
                #     )
                #     writer.flush()

                    log_metrics = {}
                    if step % cfg.gradient_accumulation_steps == 0:
                        # print_rank_0(
                        #     f"***** Evaluating policy, Epoch {epoch + 1}/{args.num_train_epochs} Step {step}/{len(prompt_train_dataloader)} *****",
                        #     args.global_rank,
                        # )
                        # perplexity = evaluation_by_ppl(trainer, ppl_eval_dataloader, device)
                        # eval_reward, eval_length, eval_kl, eval_entropy = evaluation_by_reward(
                        #     trainer, prompt_eval_dataloader, device, args, global_step, False
                        # )
                        # print_rank_0(
                        #     f"eval reward: {eval_reward} | eval length: {eval_length} | eval kl: {eval_kl} | eval entropy: {eval_entropy} | eval ppl: {perplexity}",
                        #     args.global_rank,
                        # )
                        # if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                        #     writer.add_scalar(
                        #         "eval/reward", eval_reward, global_step=global_step
                        #     )
                        #     writer.add_scalar(
                        #         "eval/length", eval_length, global_step=global_step
                        #     )
                        #     writer.add_scalar("eval/kl", eval_kl, global_step=global_step)
                        #     writer.add_scalar(
                        #         "eval/entropy", eval_entropy, global_step=global_step
                        #     )
                        #     writer.add_scalar("eval/ppl", perplexity, global_step=global_step)
                        #     writer.flush()
                        #
                        # if best_eval_reward is None:
                        #     best_eval_reward = eval_reward
                        # if eval_reward >= best_eval_reward:
                        #     best_eval_reward = eval_reward
                        #     save_model(rlhf_engine, tokenizer, args)

                        global_step += 1
                        if cfg.local_rank in [-1, 0]:
                            logs = tb_helper(clear=True)
                            log_metrics.update({f"train/{k}": v for k, v in logs.items()})
                            actor_lr = rl_engine.actor_lr_scheduler.get_lr()[0]
                            log_metrics["actor_lr"] = actor_lr

                        # Save model checkpoint
                        if cfg.save_steps > 0 and global_step % cfg.save_steps == 0:
                            # output_dir = os.path.join(cfg.output_dir, 'checkpoint-{}'.format(global_step))
                            # if cfg.local_rank in [-1, 0] and not os.path.exists(output_dir):
                            #     os.makedirs(output_dir, exist_ok=True)
                            rl_trainer.save_model(cfg.output_dir, global_step=global_step)
                            # OmegaConf.save(cfg, os.path.join(output_dir, "training_config.yaml"))
                            logger.info("Saving model checkpoint to %s", cfg.output_dir)

                        # Evaluation
                        if cfg.evaluate_during_training and cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                            # state_dict = get_state_dict(model, cfg)

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
