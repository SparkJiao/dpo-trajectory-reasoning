# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
import copy
import os
import sys
import time

import deepspeed

from omegaconf import DictConfig
import torch
import torch.nn.functional as F
import torch.distributed as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from transformers.generation.configuration_utils import GenerationConfig

from general_util.dist_utils import print_rank_0
from general_util.logger import get_child_logger
from general_util.training_utils import unwrap_model, get_zero_stage

logger = get_child_logger(__name__)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = (
                    hasattr(param, "ds_id")
                    and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            )
            with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                total += float(param.float().norm())

    return total


# Copied from https://github.com/liziniu/ReMax/blob/master/step3_rlhf_finetuning/remax_trainer.py
def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


# Copied from https://github.com/liziniu/ReMax/blob/master/step3_rlhf_finetuning/remax_trainer.py
class DeepSpeedReMaxTrainer:
    def __init__(self, rlhf_engine, cfg: DictConfig, generation_config: GenerationConfig):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.cfg = cfg
        # self.max_answer_seq_len = cfg.max_answer_seq_len
        # self.end_of_conversation_token_id = self.tokenizer(cfg.end_of_conversation_token)["input_ids"][-1]
        # self.z3_enabled = args.actor_zero_stage == 3
        # self.z3_ref_enbale = args.reference_zero_stage == 3
        self.generation_config = generation_config

        # Those value can be changed
        self.kl_ctl = cfg.kl_ctl
        self.clip_reward_value = 5.0
        self.gamma = cfg.gamma
        self.generate_time = 0.0

        # evaluation metrics
        self.eval_reward = []
        self.eval_length = []
        self.eval_kl = []
        self.eval_entropy = []

    def _generate_sequence(
            self,
            model,
            input_ids,
            attention_mask,
            global_step,
            print_answers=False,
            do_sample=True,
            synced_gpus=False,
            tag="model",
    ):
        # max_min_length = self.max_answer_seq_len + input_ids.shape[1]

        # This has been added due to a probability/nan error that happens after
        # meta-llama/Llama-2-7b-hf enabled do_sample:
        # https://huggingface.co/meta-llama/Llama-2-7b-hf/commit/6fdf2e60f86ff2481f2241aaee459f85b5b0bbb9
        # if self.actor_model.module.config.model_type == "llama":
        #     kwargs = dict(do_sample=False)
        # else:
        #     kwargs = dict()
        # kwargs = dict(
        #     do_sample=do_sample,
        #     top_p=0.9,
        #     temperature=1.0,
        # )

        generation_config = copy.deepcopy(self.generation_config)
        generation_config.do_sample = do_sample

        with torch.no_grad():
            seq = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                # max_length=max_min_length,
                # max_new_tokens=max_min_length,
                # pad_token_id=self.tokenizer.pad_token_id,
                # synced_gpus=synced_gpus,
                # **kwargs,
            )

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt
        # without supervised fine tuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = input_ids.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = ans.ne(self.tokenizer.pad_token_id).sum(dim=-1)

        if print_answers:
            logger.info(
                f"[{tag}]--- prompt --> step={global_step}, rank={dist.get_rank()}, {self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)}"
            )
            logger.info(
                f"[{tag}]--- ans    --> step={global_step}, rank={dist.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:  # if the answer is shorter than 1 token, add " ".
                seq[i, self.prompt_length] = self.tokenizer.encode(" ")[-1]
            out_seq.append(seq[i: i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim

        return out_seq

    def generate_experience(
            self, input_ids, attention_mask, labels, global_step, print_answers=False, training_mode=True
    ):
        self.eval()
        generate_start = time.time()
        seq = self._generate_sequence(
            self.actor_model.module,
            input_ids,
            attention_mask,
            global_step,
            print_answers,
            # synced_gpus=self.z3_enabled,
            do_sample=True if training_mode else False,
            synced_gpus=False,
        )
        if training_mode:
            baseline_seq = self._generate_sequence(
                self.actor_model.module,
                input_ids,
                attention_mask,
                global_step,
                print_answers,
                # synced_gpus=self.z3_enabled,
                synced_gpus=False,
                do_sample=False,
                tag="greedy",
            )
        else:
            baseline_seq = None
        generate_end = time.time()
        self.train()

        pad_token_id = self.tokenizer.pad_token_id

        action_mask = seq.not_equal(pad_token_id).long()
        if self.tokenizer.pad_token == self.tokenizer.eos_token:
            for i in range(seq.shape[0]):
                ans_mask = (seq[i, self.prompt_length:].eq(pad_token_id)).nonzero().flatten()

                if len(ans_mask) > 0:
                    # there exists an EOS token; we must set its action mask to be true.
                    # otherwise: the length may be increase
                    eos_token_pos = self.prompt_length + ans_mask[0].item()
                    action_mask[i, eos_token_pos] = 1

        if training_mode:
            baseline_action_mask = baseline_seq.not_equal(pad_token_id).long()
            if self.tokenizer.pad_token == self.tokenizer.eos_token:
                for i in range(baseline_seq.shape[0]):
                    ans_mask = (baseline_seq[i, self.prompt_length:].eq(pad_token_id)).nonzero().flatten()

                    if len(ans_mask) > 0:
                        # there exists an EOS token
                        eos_token_pos = self.prompt_length + ans_mask[0].item()
                        baseline_action_mask[i, eos_token_pos] = 1
        else:
            baseline_action_mask = None

        output = self.actor_model(seq, attention_mask=action_mask)
        with torch.no_grad():
            output_ref = self.ref_model(seq, attention_mask=action_mask)
            reward_score = self.reward_model.forward_value(
                seq, action_mask, prompt_length=self.prompt_length, labels=labels,
            )["chosen_end_scores"].detach()

            if training_mode:
                baseline_reward_score = self.reward_model.forward_value(
                    baseline_seq, baseline_action_mask, prompt_length=self.prompt_length, labels=labels,
                )["chosen_end_scores"].detach()

            values = torch.zeros_like(reward_score, device=reward_score.device)

        logits = output.logits
        logits_ref = output_ref.logits

        log_softmax_values = F.log_softmax(logits, dim=-1)
        softmax_probs = torch.exp(log_softmax_values)
        entropy = -torch.sum(softmax_probs * log_softmax_values, dim=-1)

        log_softmax_values_ref = F.log_softmax(logits_ref, dim=-1)
        full_kl = torch.sum(
            softmax_probs * (log_softmax_values - log_softmax_values_ref), dim=-1
        )

        logprobs = log_softmax_values.gather(dim=-1, index=seq[:, 1:].unsqueeze(-1)).squeeze(-1)
        ref_logprobs = log_softmax_values_ref.gather(dim=-1, index=seq[:, 1:].unsqueeze(-1)).squeeze(-1)

        self.generate_time = generate_end - generate_start

        return {
            "prompts": input_ids,
            "logprobs": logprobs,
            "ref_logprobs": ref_logprobs,
            "value": values,
            "rewards": reward_score,
            "baseline_rewards": baseline_reward_score if training_mode else None,
            "full_kl": full_kl,
            "entropy": entropy,
            "input_ids": seq,
            "attention_mask": action_mask,
        }

    def compute_returns(self, prompts, kl_divergence, reward_score, action_mask):
        returns = torch.zeros_like(kl_divergence)
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)  # + 1
        reward_clip = torch.clamp(
            reward_score, -self.clip_reward_value, self.clip_reward_value
        )
        batch_size = kl_divergence.shape[0]
        kl_ratio = 0.0
        count = 0
        for j in range(batch_size):
            cumulative_reward = reward_clip[j]
            cumulative_kl = 0
            for i in reversed(range(start, ends[j])):
                cumulative_kl = kl_divergence[j, i]

                cumulative_reward *= self.gamma
                returns[j, i] += cumulative_kl + cumulative_reward
                kl_ratio += torch.abs(cumulative_kl) / (
                        torch.abs(cumulative_reward) + torch.abs(cumulative_kl) + 1e-6
                )
                count += 1
        kl_ratio = kl_ratio / count
        return returns, kl_ratio

    def train_rl_step(self, inputs):
        # train the rlhf mode here
        prompts = inputs["prompts"]
        log_probs = inputs["logprobs"]
        ref_log_probs = inputs["ref_logprobs"]
        reward_score = inputs["rewards"]
        baseline_reward_score = inputs["baseline_rewards"]
        attention_mask = inputs["attention_mask"]
        seq = inputs["input_ids"]

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        with torch.no_grad():
            kl_divergence = -(log_probs - ref_log_probs)
            kl_divergence = self.kl_ctl * kl_divergence

            reward_score = reward_score - baseline_reward_score
            returns, kl_ratio = self.compute_returns(
                prompts, kl_divergence, reward_score, action_mask
            )

        # process the new outputs
        actor_loss = self.actor_loss_fn(
            log_probs[:, start:], returns[:, start:], action_mask[:, start:]
        )
        # return actor_loss, returns[:, start:], kl_ratio

        # Added by Fangkai: backward here so that the outside loop does not know the internal implementation, e.g., if only actor model backward or
        # both actor and critic model backward
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        # Compute outputs for logging
        outputs = {
            "actor_loss": actor_loss,
            "reward": reward_score.mean(),
            "return": returns[:, start:].mean(),
            "max_return": torch.abs(returns[:, start:]).max(),
            "kl_ratio": kl_ratio,
            "kl": torch.sum(inputs["full_kl"][:, start:] * attention_mask[:, start:]) / attention_mask[:, start:].sum(),
            "max_kl": torch.max(inputs["full_kl"][:, start:] * attention_mask[:, start:]),
            "entropy": torch.sum(inputs["entropy"][:, start:] * attention_mask[:, start:]) / attention_mask[:, start:].sum(),
        }
        return outputs

    def get_overflow(self):
        actor_overflow = self.actor_model.optimizer.overflow
        # critic_overflow = self.critic_model.optimizer.overflow

        return actor_overflow

    def actor_loss_fn(self, logprobs, returns, mask):
        # policy gradient loss
        actor_loss = torch.sum(-returns * logprobs * mask) / mask.sum()
        return actor_loss

    def _validate_training_mode(self):
        assert self.actor_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.reward_model.train()

    def eval(self):
        self.actor_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def post_process_experience(self, exp):
        # TODO: Maybe this method could be reused in `train_rl_step` function.
        # TODO: Add saving to generated content.
        reward = exp["rewards"].mean()

        prompt_length = self.prompt_length
        start = prompt_length - 1
        action_mask = exp["attention_mask"]
        answer_length = action_mask[:, start:].sum(dim=-1).float().mean()

        if "full_kl" in exp:
            kl = (
                    torch.sum(exp["full_kl"][:, start:] * action_mask[:, start:])
                    / action_mask[:, start:].sum()
            )
        else:
            kl = (
                    torch.sum(
                        (exp["logprobs"][:, start:] - exp["ref_logprobs"][:, start:])
                        * action_mask[:, start:-1]
                    )
                    / action_mask[:, start:-1].sum()
            )
        if "entropy" in exp:
            entropy = (
                    torch.sum(exp["entropy"][:, start:] * action_mask[:, start:])
                    / action_mask[:, start:].sum()
            )
        else:
            entropy = torch.zeros(1)

        self.eval_reward.append(reward.item())
        self.eval_length.append(answer_length.item())
        self.eval_kl.append(kl.item())
        self.eval_entropy.append(entropy.item())

    def get_eval_metrics(self):
        # TODO: Add saving to generated content
        eval_reward_sum = sum(self.eval_reward)
        eval_length_sum = sum(self.eval_length)
        eval_kl_sum = sum(self.eval_kl)
        eval_entropy_sum = sum(self.eval_entropy)
        eval_sample_num = len(self.eval_reward)
        if dist.is_initialized() and self.cfg.ddp_eval:
            # Gather all metrics across different ranks
            tmp = [torch.zeros((5,)) for _ in range(dist.get_world_size())]
            dist.all_gather(tmp, torch.tensor([eval_reward_sum, eval_length_sum, eval_kl_sum, eval_entropy_sum, eval_sample_num],
                                              dtype=torch.bfloat16, device=f"cuda:{self.cfg.local_rank}"))
            tmp = torch.stack(tmp, dim=0)
            eval_reward_sum = tmp[:, 0].sum().item()
            eval_length_sum = tmp[:, 1].sum().item()
            eval_kl_sum = tmp[:, 2].sum().item()
            eval_entropy_sum = tmp[:, 3].sum().item()
            eval_sample_num = tmp[:, 4].sum().item()

        metrics = {
            "reward": eval_reward_sum / eval_sample_num,
            "length": eval_length_sum / eval_sample_num,
            "kl": eval_kl_sum / eval_sample_num,
            "entropy": eval_entropy_sum / eval_sample_num,
        }
        self.eval_reward = []
        self.eval_length = []
        self.eval_kl = []
        self.eval_entropy = []
        return metrics

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        reward_model_norm = get_model_norm(self.reward_model)

        logger.warning(f"{tag} global_actor_model_norm", actor_model_norm, self.cfg.local_rank)

        logger.warning(f"{tag} global_ref_model_norm", ref_model_norm, self.cfg.local_rank)

        logger.warning(f"{tag} global_reward_model_norm", reward_model_norm, self.cfg.local_rank)

    def save_model(self, output_dir):
        # Actor model
        actor_save_dir = os.path.join(output_dir, "actor")
        unwrapped_model = unwrap_model(self.actor_model)
        self.actor_model.save_checkpoint(actor_save_dir)

        logger.info(f"Loading fp32 state dict from {actor_save_dir}")
        zero_stage = get_zero_stage(self.cfg)
        if zero_stage == 3:
            state_dict = self.actor_model._zero3_consolidated_16bit_state_dict()
        elif zero_stage == 2:
            state_dict = get_fp32_state_dict_from_zero_checkpoint(actor_save_dir)
        else:
            state_dict = unwrapped_model.state_dict()

        if self.cfg.local_rank in [-1, 0]:
            unwrapped_model.save_pretrained(actor_save_dir, state_dict=state_dict)

            self.tokenizer.save_pretrained(actor_save_dir)


class DeepSpeedReMaxTrainerUnsupervised(DeepSpeedReMaxTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
