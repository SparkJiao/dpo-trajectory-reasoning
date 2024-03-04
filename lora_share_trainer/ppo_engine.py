# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
# Nanyang Technological University, Singapore (NTU, Singapore)
# Fangkai Jiao

import os
import time

from typing import Union, Callable

import deepspeed
import fairscale.nn.model_parallel.initialize as mpu
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.generation import GenerationConfig

from general_util import training_utils
from general_util.dist_utils import print_rank_0
from general_util.logger import get_child_logger
from general_util.training_utils import get_zero_stage
from general_util.transformer_engine import convert_model
from lora_share_trainer.utils.fp8 import fp8_func_wrap
from lora_share_trainer.utils.utils import log_init, gather_log_probs, trainer_save_single_model
from lora_share_trainer.utils.post_process import react_process_reward

logger = get_child_logger(__name__)

try:
    import transformer_engine.pytorch as transformer_engine
    from transformer_engine.common import recipe
except ImportError:
    logger.info("Transformer Engine package is missing, skipping tests")


class DeepSpeedChatPPOEngine:
    def __init__(self,
                 cfg: DictConfig,
                 actor_model: PreTrainedModel,
                 ref_model: PreTrainedModel,
                 critic_model: PreTrainedModel,
                 reward_model: PreTrainedModel,
                 tokenizer: Union[str, PreTrainedTokenizer],
                 actor_fp8: bool = False,
                 critic_fp8: bool = False,
                 reward_fp8: bool = False,
                 reference_fp8: bool = False,
                 ):
        self.cfg = cfg
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.actor_fp8 = actor_fp8
        self.critic_fp8 = critic_fp8
        self.reward_fp8 = reward_fp8
        self.reference_fp8 = reference_fp8

        if self.actor_fp8:
            convert_model(actor_model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True)
        if self.critic_fp8:
            convert_model(critic_model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True)
        if self.reward_fp8:
            convert_model(reward_model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True)
        if self.reference_fp8:
            convert_model(ref_model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True)

        self.actor, self.actor_optim, self.actor_lr_scheduler = self._init_actor(actor_model)
        self.ref = self._init_ref(ref_model)
        self.actor_ema = None
        if self.cfg.enable_ema:
            self.actor_ema = self._init_ema(actor_model)

        self.critic, self.critic_optim, self.critic_lr_scheduler = self._init_critic(critic_model)
        self.reward = self._init_reward(reward_model)

        assert self.actor != self.ref
        assert self.critic != self.reward

    def _init_actor(self, actor_model: PreTrainedModel, ):
        stime = log_init("Actor")

        ds_config = self.cfg.actor_ds_config
        if "total_num_steps" in ds_config.scheduler.params:
            ds_config.scheduler.params.total_num_steps = self.cfg.max_steps
        ds_config.scheduler.params.warmup_num_steps = self.cfg.warmup_steps
        ds_config = OmegaConf.to_container(ds_config, resolve=True)
        ds_config["train_mirco_batch_size_per_gpu"] = self.cfg.per_gpu_train_batch_size
        # ds_config["train_batch_size"] = self.cfg.per_gpu_train_batch_size * self.cfg.gradient_accumulation_steps_actor

        optim_params = training_utils.get_optimizer_grouped_parameters(actor_model, self.cfg.actor_weight_decay)

        actor_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=actor_model,
            model_parameters=optim_params,
            config_params=ds_config,
            mpu=mpu if mpu.model_parallel_is_initialized() else None,
        )

        log_init("Actor", stime)

        return actor_engine, optimizer, scheduler

    def _init_ref(self, actor_model_or_reward_model: PreTrainedModel):
        stime = log_init("Reference")

        ds_config = self.cfg.ref_ds_config
        if ds_config:  # In case we need ZeRO-3 inference.
            # if "total_num_steps" in ds_config.scheduler.params:
            #     ds_config.scheduler.params.total_num_steps = self.cfg.max_steps
            # ds_config.scheduler.params.warmup_num_steps = self.cfg.warmup_steps
            ds_config = OmegaConf.to_container(ds_config, resolve=True)
            ds_config["train_mirco_batch_size_per_gpu"] = self.cfg.per_gpu_train_batch_size
            # ds_config["train_batch_size"] = self.cfg.per_gpu_train_batch_size * self.cfg.gradient_accumulation_steps_ref

            ref_engine, *_ = deepspeed.initialize(
                model=actor_model_or_reward_model,
                config_params=ds_config,
                mpu=mpu if mpu.model_parallel_is_initialized() else None,
            )
        else:
            ref_engine = actor_model_or_reward_model
            ref_engine.to(device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        log_init("Reference", stime)

        return ref_engine

    def _init_ema(self, actor_model: PreTrainedModel, ):
        stime = log_init("EMA")

        ds_config = self.cfg.actor_ds_config
        # if "total_num_steps" in ds_config.scheduler.params:
        #     ds_config.scheduler.params.total_num_steps = self.cfg.max_steps
        # ds_config.scheduler.params.warmup_num_steps = self.cfg.warmup_steps
        if ds_config.zero_optimization.stage != 3:
            ds_config.zero_optimization.stage = 0

        ds_config = OmegaConf.to_container(ds_config, resolve=True)
        ds_config["train_mirco_batch_size_per_gpu"] = self.cfg.per_gpu_train_batch_size
        # ds_config["train_batch_size"] = self.cfg.per_gpu_train_batch_size * self.cfg.gradient_accumulation_steps_actor
        if "optimizer" in ds_config:
            ds_config.pop("optimizer")
        if "scheduler" in ds_config:
            ds_config.pop("scheduler")

        actor_engine, *_ = deepspeed.initialize(
            model=actor_model,
            config_params=ds_config,
            mpu=mpu if mpu.model_parallel_is_initialized() else None,
        )

        log_init("EMA", stime)

        return actor_engine

    def _init_critic(self, critic_model: PreTrainedModel):
        stime = log_init("Critic")

        ds_config = self.cfg.critic_ds_config
        if "total_num_steps" in ds_config.scheduler.params:
            ds_config.scheduler.params.total_num_steps = self.cfg.max_steps
        ds_config.scheduler.params.warmup_num_steps = self.cfg.warmup_steps
        ds_config = OmegaConf.to_container(ds_config, resolve=True)
        ds_config["train_mirco_batch_size_per_gpu"] = self.cfg.per_gpu_train_batch_size
        # ds_config["train_batch_size"] = self.cfg.per_gpu_train_batch_size * self.cfg.gradient_accumulation_steps_actor

        optim_params = training_utils.get_optimizer_grouped_parameters(critic_model, self.cfg.critic_weight_decay)

        critic_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=critic_model,
            model_parameters=optim_params,
            config_params=ds_config,
            mpu=mpu if mpu.model_parallel_is_initialized() else None,
        )

        log_init("Critic", stime)
        return critic_engine, optimizer, scheduler

    def _init_reward(self, reward_model: PreTrainedModel):
        stime = log_init("Reward")

        ds_config = self.cfg.rm_ds_config
        if ds_config:  # In case we need ZeRO-3 inference.
            # if "total_num_steps" in ds_config.scheduler.params:
            #     ds_config.scheduler.params.total_num_steps = self.cfg.max_steps
            # ds_config.scheduler.params.warmup_num_steps = self.cfg.warmup_steps
            ds_config = OmegaConf.to_container(ds_config, resolve=True)
            ds_config["train_mirco_batch_size_per_gpu"] = self.cfg.per_gpu_train_batch_size
            # ds_config["train_batch_size"] = self.cfg.per_gpu_train_batch_size * self.cfg.gradient_accumulation_steps_ref

            ref_engine, *_ = deepspeed.initialize(
                model=reward_model,
                config_params=ds_config,
                mpu=mpu if mpu.model_parallel_is_initialized() else None,
            )
        else:
            ref_engine = reward_model
            ref_engine.to(device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        log_init("Reward", stime)

        return ref_engine


class DSChatPPOTrainer:
    def __init__(self, engine: DeepSpeedChatPPOEngine, cfg: DictConfig,
                 reward_post_fn: Callable = react_process_reward(),
                 generation_config: GenerationConfig = GenerationConfig(
                     max_new_tokens=2048,
                     do_sample=True,
                     num_return_sequences=1,
                 ),
                 ):
        self.engine = engine
        self.cfg = cfg
        self.actor_model = engine.actor
        self.ref_model = engine.ref
        self.critic_model = engine.critic
        self.reward_model = engine.reward
        self.tokenizer = engine.tokenizer

        # FP8 setting
        self.actor_fp8 = engine.actor_fp8
        self.critic_fp8 = engine.critic_fp8
        self.reward_fp8 = engine.reward_fp8
        self.reference_fp8 = engine.reference_fp8

        # Those value can be changed
        self.kl_ctl = getattr(cfg, "kl_ctl", 0.1)
        self.clip_reward_value = getattr(cfg, "clip_reward_value", 5)
        self.clip_range = getattr(cfg, "clip_range", 0.2)
        self.clip_range_value = getattr(cfg, "clip_range_value", 0.2)
        self.gamma = getattr(cfg, "gamma", 1.0)
        self.lam = getattr(cfg, "lam", 0.95)
        self.generate_time = getattr(cfg, "generate_time", 1)

        # EMA
        self.enable_ema = cfg.enable_ema

        self.reward_post_fn = reward_post_fn

        # Defaults. To be verified.
        self.z3_enabled = False
        self.compute_fp32_loss = True
        self.generation_config = generation_config

        # scaling recipe
        if self.actor_fp8 or self.critic_fp8:
            self.fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
                                                    amax_history_len=getattr(cfg, "fp8_amax_history_len", 16),
                                                    amax_compute_algo=getattr(cfg, "fp8_amax_compute_algo", "max"))
        else:
            self.fp8_recipe = None

    def _generate_sequence(self, prompt_input_ids, attention_mask, step):

        # max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # if self.actor_model.module.config.model_type == "llama":
        #     kwargs = dict(do_sample=False)
        # else:
        #     kwargs = dict()
        #
        with torch.no_grad():
            # seq = self.actor_model.module.generate(
            #     prompt_input_ids,
            #     attention_mask=attention_mask,
            #     generation_config=self.generation_config,
            #     synced_gpus=self.z3_enabled,
            # #    **kwargs)
            # )
            seq = fp8_func_wrap(self.actor_model.module.generate, self.actor_fp8, self.fp8_recipe,
                                prompt_input_ids,
                                attention_mask=attention_mask,
                                generation_config=self.generation_config,
                                synced_gpus=self.z3_enabled,
                                )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompt_input_ids.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        if self.cfg.print_answers and step % 5 == 0:
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return {"seq": out_seq}

    def generate_experience(self, input_ids, attention_mask, global_step, **kwargs):
        prompt_input_ids = input_ids
        mask = attention_mask

        self.eval()
        generate_start = time.time()
        seq = self._generate_sequence(prompt_input_ids, mask, global_step)["seq"]  # `seq` should have special tokens.
        generate_end = time.time()
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        with torch.no_grad():
            output = fp8_func_wrap(self.actor_model, self.actor_fp8, self.fp8_recipe, seq, attention_mask)
            output_ref = fp8_func_wrap(self.ref_model, self.reference_fp8, self.fp8_recipe, seq, attention_mask)
            output_rm = fp8_func_wrap(self.reward_model, self.reward_fp8, self.fp8_recipe, seq, attention_mask)
            reward_score = self.reward_post_fn(prompt_input_ids, seq, output_rm, self.tokenizer, **kwargs).detach()
            values = fp8_func_wrap(self.critic_model, self.critic_fp8, self.fp8_recipe, seq, attention_mask)["values"].detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start

        return {
            'prompt_input_ids': prompt_input_ids,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:], self.actor_model.module.config.pad_token_id),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:], self.actor_model.module.config.pad_token_id),
            'values': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }

    def compute_rewards(self, prompt_input_ids, log_probs, ref_log_probs, reward_score, action_mask):
        """

        :param prompt_input_ids:
        :param log_probs:
        :param ref_log_probs:
        :param reward_score: shares the same length with `values`: n.
        :param action_mask: length: n - 1, already shifted to the right by 1.
        :return:
        """
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)  # Actual length of full conversation - 1
        rewards = kl_divergence_estimate  # [batch, n - 1]
        assert rewards.shape[1] == reward_score.shape[1] - 1
        start = prompt_input_ids.shape[1] - 1  # The log_probs and ref_log_probs are calculated from the second token.
        ends = start + action_mask[:, start:].sum(1) + 1  # `ends` demonstrate the actual length of the conversation plus 1.
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)  # The last token should be the `eos` token.  # [batch(, n)]
        batch_size = log_probs.shape[0]
        # for j in range(batch_size):
        #     if len(reward_clip.size()) > 1:
        #         # rewards[j, start:ends[j]] += reward_clip[j][1:]  # TODO: I'm not sure for process rewards, it should be reward_clip[1:] or reward_clip[:-1].
        #         # [start, ends[j]] is the response part in `rewards` because the log_probs and ref_log_probs are calculated from the second token.
        #         # The maximum value of `ends` is bigger than the length of `rewards` with 1.
        #         # [start + 1, ends[j]] is the response part in the full sequence, e.g., `reward_clip` and `values`.
        #         # `reward_clip` are the rewards assigned to each token that can be obtained only when they are really being approached.
        #         rewards[j, start:(ends[j] - 1)] += reward_clip[j, (start + 1):ends[j]]
        #     else:
        #         # rewards[j, start:ends[j]][-1] += reward_clip[j]
        #         rewards[j, start:(ends[j] - 1)][-1] += reward_clip[j]
        if len(reward_clip.size()) > 1:
            reward_clip = reward_clip[:, 1:]
            reward_clip[:, :start] = 0
            reward_clip[~(action_mask.bool())] = 0
            rewards += reward_clip
        else:  # [batch]
            rewards[:, ends - 1] += reward_clip

        return rewards

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        last_gae_lamda = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            next_values = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_values - values[:, t]
            last_gae_lamda = delta + self.gamma * self.lam * last_gae_lamda
            advantages_reversed.append(last_gae_lamda)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        # policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        # value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.clip_range_value,
            old_values + self.clip_range_value,
        )
        if self.compute_fp32_loss:
            values = values.float()
            values_clipped = values_clipped.float()
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def train_rl_step(self, inputs):
        # train the rlhf mode here
        # process the old outputs
        prompts = inputs['prompt_input_ids']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['values']
        seq = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        start = prompts.size()[-1] - 1  # We have shifted the generated sequence to the right by 1.
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            # for i in range(old_rewards.shape[0]):
            # old_rewards[i, ends[i]:] = 0  # TODO: Should we also modify this to correspond to line 443?
            # old_values[i, ends[i]:] = 0
            # old_rewards[i, (ends[i] - 1):] = 0
            # old_values[i, (ends[i] - 1):] = 0
            old_rewards[~(action_mask.bool())] = 0
            old_values[~(action_mask.bool())] = 0  # Since the advantages are computed since `start`, we do not need to mask the response inputs here.
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

        # process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = fp8_func_wrap(self.actor_model, self.actor_fp8, self.fp8_recipe, **batch).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:], self.actor_model.module.config.pad_token_id)
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.cfg.align_overflow:
            self.actor_model.step()

        value = fp8_func_wrap(self.critic_model, self.critic_fp8, self.fp8_recipe, **batch)["values"][:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:, start:], returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)

        if self.cfg.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        if len(reward_score.size()) > 1:
            log_reward = reward_score.sum(dim=1).mean()
        else:
            log_reward = reward_score.mean()
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "return": returns.mean(),
            "reward": log_reward,
        }

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.cfg.torch_dtype == torch.bfloat16:
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow
        critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow, critic_overflow

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def save_model(self, output_dir, global_step: int = -1):
        if global_step != -1:
            actor_save_dir = os.path.join(output_dir, "actor", "checkpoint-{}".format(global_step))
            critic_save_dir = os.path.join(output_dir, "critic", "checkpoint-{}".format(global_step))
        else:
            actor_save_dir = os.path.join(output_dir, "actor")
            critic_save_dir = os.path.join(output_dir, "critic")

        if self.cfg.local_rank in [-1, 0]:
            if not os.path.exists(actor_save_dir):
                os.makedirs(actor_save_dir, exist_ok=True)
            if not os.path.exists(critic_save_dir):
                os.makedirs(critic_save_dir, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

        # Actor model
        trainer_save_single_model(self.actor_model,
                                  actor_save_dir,
                                  self.cfg.local_rank,
                                  get_zero_stage(self.cfg.actor_ds_config),
                                  tokenizer=self.tokenizer,
                                  save_ds_state=self.cfg.save_ds_state,
                                  state_save_dir=os.path.join(output_dir, "actor"),
                                  )

        # Critic model
        trainer_save_single_model(self.critic_model,
                                  critic_save_dir,
                                  self.cfg.local_rank,
                                  get_zero_stage(self.cfg.critic_ds_config),
                                  save_ds_state=self.cfg.save_ds_state,
                                  tokenizer=self.tokenizer,
                                  state_save_dir=os.path.join(output_dir, "critic"),
                                  )

        if self.cfg.enable_ema:
            if global_step != -1:
                ema_save_dir = os.path.join(output_dir, "actor_ema", "checkpoint-{}".format(global_step))
            else:
                ema_save_dir = os.path.join(output_dir, "actor_ema")
            if self.cfg.local_rank in [-1, 0]:
                if not os.path.exists(ema_save_dir):
                    os.makedirs(ema_save_dir, exist_ok=True)

            if dist.is_initialized():
                dist.barrier()

            # EMA model
            trainer_save_single_model(self.engine.actor_ema,
                                      ema_save_dir,
                                      self.cfg.local_rank,
                                      get_zero_stage(self.cfg.actor_ds_config),
                                      save_ds_state=self.cfg.save_ds_state,
                                      tokenizer=self.tokenizer,
                                      state_save_dir=os.path.join(output_dir, "actor_ema"),
                                      )

        if self.cfg.local_rank in [-1, 0]:
            OmegaConf.save(self.cfg, os.path.join(output_dir, "training_config.yaml"))

    def resume(self, checkpoint_dir):
        self.actor_model.load_checkpoint(checkpoint_dir)
        self.critic_model.load_checkpoint(checkpoint_dir)
