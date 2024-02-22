import os
from logging import Logger
from typing import Optional, Union, Tuple, List, Callable

import hydra
import omegaconf
import torch
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers.models.gemma.modeling_gemma import (
    GemmaForCausalLM as HfGemmaForCausalLM,
    PreTrainedModel,
    CausalLMOutputWithPast,
    GemmaModel,
    GemmaPreTrainedModel,
    GemmaConfig,
)

from general_util.logger import get_child_logger
from general_util.training_utils import get_rank
from models.utils import DPOModelOutput, RewardModelOutput
from models.llama import PreTrainedModelPeftMixin

logger: Logger = get_child_logger(__name__)

REFERENCE_MODEL: HfGemmaForCausalLM


def return_single_device_map():
    return {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}


class PreTrainedGemmaPeftMixin(PreTrainedModelPeftMixin):
    pass


# Copied from models.llama.py
def llama_dpo_batch_forward(model: HfGemmaForCausalLM, input_ids: torch.LongTensor, attention_mask: torch.Tensor, labels: torch.LongTensor):
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)
    logits = logits.float()

    labels = labels[:, 1:].clone()

    loss_mask = labels.ne(model.config.pad_token_id)
    labels[~loss_mask] = 0

    per_token_logprobs = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return logits, (per_token_logprobs * loss_mask).sum(-1), loss_mask


def llama_last_token_cls_batch_forward(model: GemmaModel, linear: nn.Linear,
                                       input_ids: torch.LongTensor, attention_mask: torch.Tensor,
                                       pad_token_id: int, return_full_logits: bool = False):
    transformer_outputs = model(
        input_ids,
        attention_mask=attention_mask,
    )
    hidden_states = transformer_outputs[0]

    batch_size = input_ids.shape[0]
    sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(device=hidden_states.device)

    if return_full_logits:
        rewards = linear(hidden_states)
        return rewards, sequence_lengths

    last_token_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
    rewards = linear(last_token_states)
    return rewards, sequence_lengths


def llama_last_token_forward_value(model: GemmaModel, linear: nn.Linear,
                                   input_ids: torch.LongTensor, attention_mask: torch.Tensor,
                                   pad_token_id: int):
    transformer_outputs = model(
        input_ids,
        attention_mask=attention_mask,
    )
    hidden_states = transformer_outputs[0]

    batch_size = input_ids.shape[0]
    sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(device=hidden_states.device)

    values = linear(hidden_states)
    rewards = values[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
    return values, rewards, sequence_lengths


class GemmaForCausalLMDPO(PreTrainedGemmaPeftMixin, HfGemmaForCausalLM):
    def __init__(self, config, beta: float = 0.1, label_smoothing: float = 0.0, use_ipo: bool = False, loss_type: str = "sigmoid"):
        super().__init__(config)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.use_ipo = use_ipo
        self.loss_type = loss_type
        logger.warning(f"Using loss type: {self.loss_type}")

        # Initialize weights and apply final processing
        self.post_init()

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.use_ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "sigmoid":
            log_sigmoid = nn.LogSigmoid()
            losses = -log_sigmoid(self.beta * logits) * (1 - self.label_smoothing) - log_sigmoid(-self.beta * logits) * self.label_smoothing
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses.mean(), chosen_rewards, rejected_rewards

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        half = input_ids.size(0) // 2

        policy_logits, policy_logprobs, policy_loss_mask = llama_dpo_batch_forward(self, input_ids, attention_mask, labels)
        with torch.no_grad():
            ref_logits, ref_logprobs, ref_loss_mask = llama_dpo_batch_forward(REFERENCE_MODEL, input_ids, attention_mask, labels)

        policy_chosen_logits, policy_reject_logits = policy_logits[:half], policy_logits[half:]
        policy_chosen_logprobs, policy_reject_logprobs = policy_logprobs[:half], policy_logprobs[half:]

        # ref_chosen_logits, ref_reject_logits = ref_logits[:half], ref_logits[half:]
        ref_chosen_logprobs, ref_reject_logprobs = ref_logprobs[:half], ref_logprobs[half:]

        loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps=policy_chosen_logprobs,
            policy_rejected_logps=policy_reject_logprobs,
            reference_chosen_logps=ref_chosen_logprobs,
            reference_rejected_logps=ref_reject_logprobs,
            reference_free=False,
        )

        return DPOModelOutput(
            loss=loss,
            chosen_reward=chosen_rewards.mean(),
            rejected_reward=rejected_rewards.mean(),
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_reject_logits,
        )

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            save_peft_format: bool = True,
            **kwargs,
    ):
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token,
                                **kwargs)

        if is_main_process:
            config = self.config
            config.architectures = ["LlamaForCausalLM"]
            config.save_pretrained(save_directory)
            logger.warning("Config architecture is override to LlamaForCausalLM")


class GemmaRewardModel(PreTrainedGemmaPeftMixin, GemmaPreTrainedModel):
    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.model = GemmaModel(config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def pair_wise_loss(self,
                       chosen_rewards: torch.FloatTensor,
                       rejected_rewards: torch.FloatTensor, ):
        reward_loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        return reward_loss

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        half = input_ids.size(0) // 2

        rewards, sequence_lengths = llama_last_token_cls_batch_forward(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id)
        chosen_rewards, rejected_rewards = rewards[:half], rewards[half:]

        loss = self.pair_wise_loss(chosen_rewards, rejected_rewards)

        return DPOModelOutput(
            loss=loss,
            chosen_reward=chosen_rewards.mean(),
            rejected_reward=rejected_rewards.mean(),
            policy_chosen_logits=None,
            policy_rejected_logits=None,
            batch_chosen_reward=chosen_rewards,
            batch_rejected_reward=rejected_rewards,
        )


class GemmaRewardModelForEval(GemmaRewardModel):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        rewards, sequence_lengths = llama_last_token_cls_batch_forward(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id)
        return DPOModelOutput(
            batch_chosen_reward=rewards,
        )


class GemmaModelForSequenceClassification(PreTrainedGemmaPeftMixin, GemmaPreTrainedModel):
    def __init__(self, config: GemmaConfig):
        super().__init__(config)
        self.model = GemmaModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            values: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        half = input_ids.size(0) // 2

        rewards, sequence_lengths = llama_last_token_cls_batch_forward(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(rewards, values)

        return DPOModelOutput(
            loss=loss,
            logits=rewards,
        )


class GemmaModelForSequenceClassificationForEval(GemmaModelForSequenceClassification):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        rewards, sequence_lengths = llama_last_token_cls_batch_forward(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id,
                                                                       return_full_logits=True)
        return DPOModelOutput(
            logits=rewards,
        )


class GemmaModelForSequenceClassificationForRL(GemmaModelForSequenceClassification):
    def __init__(self, config: GemmaConfig, reduction_ids: List[int]):
        super().__init__(config)

        if isinstance(reduction_ids, omegaconf.ListConfig):
            reduction_ids = list(reduction_ids)

        self.reduction_ids = reduction_ids

    def logit2prob(self, logits):
        probs = torch.softmax(logits, dim=-1)
        if len(logits.size()) == 3:
            probs = probs[:, :, self.reduction_ids].sum(dim=-1)
        elif len(logits.size()) == 2:
            probs = probs[:, self.reduction_ids].sum(dim=-1)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.size()}")
        return probs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[Tuple, RewardModelOutput]:
        values, rewards, sequence_lengths = llama_last_token_forward_value(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id)
        values = self.logit2prob(values)
        rewards = self.logit2prob(rewards)
        return RewardModelOutput(
            values=values,
            chosen_end_scores=rewards,
            sequence_lengths=sequence_lengths,
        )


class GemmaForCausalLM(PreTrainedGemmaPeftMixin, HfGemmaForCausalLM):
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_labels[shift_labels.eq(self.config.pad_token_id)] = -100
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
