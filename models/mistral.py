import os
from logging import Logger
from typing import Optional, Union, Tuple, List, Callable

import hydra
import omegaconf
import torch
from torch import nn
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
    get_peft_model,
)

from transformers.models.mistral import (
    MistralConfig,
    MistralModel,
    MistralPreTrainedModel,
    MistralForCausalLM as HfMistralForCausalLM,
    MistralForSequenceClassification
)

from general_util.logger import get_child_logger
from general_util.training_utils import get_rank
from models.llama import PreTrainedModelPeftMixin
from models.utils import DPOModelOutput

logger: Logger = get_child_logger(__name__)

REFERENCE_MODEL: HfMistralForCausalLM


def llama_last_token_cls_batch_forward(model: MistralModel, linear: nn.Linear,
                                       input_ids: torch.LongTensor, attention_mask: torch.Tensor,
                                       pad_token_id: int, ):
    transformer_outputs = model(
        input_ids,
        attention_mask=attention_mask,
    )
    hidden_states = transformer_outputs[0]

    batch_size = input_ids.shape[0]
    sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(device=hidden_states.device)
    last_token_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
    rewards = linear(last_token_states)
    return rewards, sequence_lengths


class MistralRewardModel(PreTrainedModelPeftMixin, MistralPreTrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.model = MistralModel(config)
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


class MistralRewardModelForEval(MistralRewardModel):
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
