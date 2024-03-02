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
from lora_share_trainer.ppo_engine import DSChatPPOTrainer
from lora_share_trainer.utils import ds_utils

logger = get_child_logger(__name__)

try:
    import transformer_engine.pytorch as transformer_engine
    from transformer_engine.common import recipe
except ImportError:
    logger.info("Transformer Engine package is missing, skipping tests")


class GRPOEngine:
    def __init__(self,
                 cfg: DictConfig,
                 actor_model: PreTrainedModel,
                 ref_model: PreTrainedModel,
                 reward_model: PreTrainedModel,
                 tokenizer: Union[str, PreTrainedTokenizer],
                 actor_fp8: bool = False,
                 reward_fp8: bool = False,
                 reference_fp8: bool = False,
                 ):
        self.cfg = cfg
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.actor_fp8 = actor_fp8
        self.reward_fp8 = reward_fp8
        self.reference_fp8 = reference_fp8

        if self.actor_fp8:
            convert_model(actor_model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True)
        if self.reward_fp8:
            convert_model(reward_model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True)
        if self.reference_fp8:
            convert_model(ref_model, to_transformer_engine=True, _convert_linear=True, _convert_ln=True)

        self.actor, self.actor_optim, self.actor_lr_scheduler = self._init_actor(actor_model)
        self.ref = self._init_ref(ref_model)
        self.actor_ema = None
        if self.cfg.enable_ema:
            self.actor_ema = self._init_ema(actor_model)

        self.reward = self._init_reward(reward_model)

        assert self.actor != self.ref

    def _init_actor(self, actor_model: PreTrainedModel, ):
        stime = log_init("Actor")

        actor_engine, optimizer, scheduler = ds_utils.init_ds_training_engine(actor_model, self.cfg.actor_ds_config, self.cfg)

        log_init("Actor", stime)

        return actor_engine, optimizer, scheduler

    def _init_ref(self, actor_model_or_reward_model: PreTrainedModel):
        stime = log_init("Reference")

        if self.cfg.ref_ds_config:  # In case we need ZeRO-3 inference.
            ref_engine, *_ = ds_utils.init_ds_eval_engine(actor_model_or_reward_model, self.cfg.ref_ds_config, self.cfg)
        else:
            ref_engine = actor_model_or_reward_model
            ref_engine.to(device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        log_init("Reference", stime)

        return ref_engine

    def _init_ema(self, actor_model: PreTrainedModel, ):
        stime = log_init("EMA")

        actor_ema_engine = ds_utils.init_ds_eval_engine(actor_model, self.cfg.actor_ds_config, self.cfg)

        log_init("EMA", stime)

        return actor_ema_engine

    def _init_reward(self, reward_model: PreTrainedModel):
        stime = log_init("Reward")

        if self.cfg.rm_ds_config:  # In case we need ZeRO-3 inference.
            reward_engine = ds_utils.init_ds_eval_engine(reward_model, self.cfg.rm_ds_config, self.cfg)
        else:
            reward_engine = reward_model
            reward_engine.to(device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        log_init("Reward", stime)

        return reward_engine


class GRPOTrainer:
    def __init__(self, engine: GRPOEngine, cfg: DictConfig,
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
        self.reward_model = engine.reward
        self.tokenizer = engine.tokenizer

        # FP8 setting
        self.actor_fp8 = engine.actor_fp8
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
        self.num_return_sequences = generation_config.num_return_sequences

        # scaling recipe
        if self.actor_fp8:
            self.fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.HYBRID,
                                                    amax_history_len=getattr(cfg, "fp8_amax_history_len", 16),
                                                    amax_compute_algo=getattr(cfg, "fp8_amax_compute_algo", "max"))
        else:
            self.fp8_recipe = None

    def _generate_sequence(self, prompt_input_ids, attention_mask, step):
        with torch.no_grad():
            seq = fp8_func_wrap(self.actor_model.module.generate, self.actor_fp8, self.fp8_recipe,
                                prompt_input_ids,
                                attention_mask=attention_mask,
                                generation_config=self.generation_config,
                                synced_gpus=self.z3_enabled,
                                )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised fine-tuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompt_input_ids.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=1)

        if self.cfg.print_answers and step % 5 == 0:
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompt_input_ids, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        # Commented by Fangkai: Temporarily disable this filtering due to the difficulty in generation padding for reward computing.
        # out_seq = []
        # for i in range(batch_size):
        #     if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, drop it
        #         continue
        #     else:
        #         out_seq.append(seq[i:i + 1])
        # out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return {"seq": seq}

    def unpack(self, x):
        return x.reshape(-1, self.num_return_sequences, *x.size()[1:])

    @staticmethod
    def pack(x):
        return x.reshape(-1, *x.size()[2:])

    def generate_experience(self, input_ids, attention_mask, global_step):
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
            reward_score = self.reward_post_fn(prompt_input_ids, seq, output_rm, self.tokenizer).detach()
            # print(reward_score.size())  # [16, 1070]

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        self.generate_time = generate_end - generate_start

        return {
            'prompt_input_ids': prompt_input_ids,
            'logprobs': self.unpack(gather_log_probs(logits[:, :-1, :], seq[:, 1:], self.actor_model.module.config.pad_token_id)),
            'ref_logprobs': self.unpack(gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:], self.actor_model.module.config.pad_token_id)),
            'rewards': self.unpack(reward_score),
            'input_ids': self.unpack(seq),
            "attention_mask": self.unpack(attention_mask),
        }

    def compute_rewards(self, reward_score):
        """
        :param reward_score: shares the same length with `values`: n. [batch * num_return_sequences, (sequence_length)]
        :return: normalized_reward: [batch, num_return_sequences, (sequence_length)]
        """
        reward_score = reward_score.reshape(-1, self.num_return_sequences, reward_score.size(-1))
        bsz, _, seq_len = reward_score.size()

        if self.clip_reward_value > 0:
            reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        else:
            reward_clip = reward_score

        dims = list(range(1, len(reward_clip.size())))

        if len(reward_clip.size()) in [2, 3]:  # dim=3 for process rewards and dim=2 for outcome rewards.
            step_mask = reward_clip.ne(0)

            reward_avg = reward_clip.sum(dim=dims) / step_mask.sum(dim=dims)

            reward_std = (reward_clip - reward_avg[:, None, None]).pow(2)
            reward_std[~step_mask] = 0
            reward_std = reward_std.sum(dim=dims) / step_mask.sum(dim=dims)
            reward_std = reward_std.sqrt()

            normalized_reward = reward_clip - reward_avg[:, None, None]
            normalized_reward = normalized_reward / (reward_std[:, None, None] + 1e-8)
            normalized_reward[~step_mask] = 0
        else:
            raise ValueError(f"Unknown reward_clip size: {reward_clip.size()}")

        return normalized_reward

    @staticmethod
    def get_advantages_and_returns(normalized_reward):
        if len(normalized_reward.size()) == 3:
            # Compute reverse accumulated rewards
            advantages = torch.cumsum(normalized_reward.flip(dims=(2,)), dim=2).flip(dims=(2,))  # [batch, num_return_sequences, sequence_length]
        elif len(normalized_reward.size()) == 2:
            advantages = normalized_reward  # [batch, num_return_sequences]
        else:
            raise ValueError()
        return advantages.detach()

    def actor_loss_fn(self, logprobs, old_logprobs, ref_logprobs, advantages, mask):
        # policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()

        log_ratio = (ref_logprobs - logprobs) * mask
        ratio = torch.exp(log_ratio)
        kl_loss = -self.kl_ctl * torch.sum((ratio - log_ratio - 1) * mask) / mask.sum()

        return pg_loss + kl_loss

    def train_rl_step(self, inputs):
        # train the rlhf mode here
        # process the old outputs
        prompts = inputs['prompt_input_ids']
        log_probs = self.pack(inputs['logprobs'])
        ref_log_probs = self.pack(inputs['ref_logprobs'])
        reward_score = self.pack(inputs['rewards'])
        seq = self.pack(inputs['input_ids'])
        attention_mask = self.pack(inputs['attention_mask'])

        start = prompts.size()[-1] - 1  # We have shifted the generated sequence to the right by 1.
        action_mask = attention_mask[:, 1:]

        with torch.no_grad():
            old_rewards = self.compute_rewards(reward_score)  # already masked in reward post-processing so we do not mask again here.
            advantages = self.get_advantages_and_returns(old_rewards)
            if len(advantages.size()) == 3:
                advantages = advantages[:, :, 1:]
                advantages[~self.unpack(action_mask).bool()] = 0
                advantages = advantages[:, :, start:]
            advantages = advantages.reshape(-1, advantages.size(-1))

        # process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = fp8_func_wrap(self.actor_model, self.actor_fp8, self.fp8_recipe, **batch).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:], self.actor_model.module.config.pad_token_id)
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], ref_log_probs[:, start:], advantages, action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.cfg.align_overflow:
            self.actor_model.step()

        if self.cfg.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow:
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        if len(reward_score.size()) > 1:
            log_reward = reward_score.sum(dim=1).mean()
        else:
            log_reward = reward_score.mean()
        return {
            "actor_loss": actor_loss,
            "reward": log_reward,
        }

    def get_overflow(self):
        # Overflow is not expected when using bf16
        # Therefore, DeepSpeed's BF16_Optimizer does not maintain an overflow indication
        if self.cfg.torch_dtype == torch.bfloat16:
            return False, False

        actor_overflow = self.actor_model.optimizer.overflow

        return actor_overflow

    def train(self):
        self.actor_model.train()

    def eval(self):
        self.actor_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def save_model(self, output_dir, global_step: int = -1):
        if global_step != -1:
            actor_save_dir = os.path.join(output_dir, "actor", "checkpoint-{}".format(global_step))
        else:
            actor_save_dir = os.path.join(output_dir, "actor")

        if self.cfg.local_rank in [-1, 0]:
            if not os.path.exists(actor_save_dir):
                os.makedirs(actor_save_dir, exist_ok=True)

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
