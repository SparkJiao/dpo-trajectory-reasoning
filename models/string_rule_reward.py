import re

import torch
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict

from models.reward_model_mixin import RewardModelMixin, RewardModelOutputs


class MultipleChoiceAccuracyReward(nn.Module, RewardModelMixin):
    def __init__(self, base_model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.option2int = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

    def forward(self, *args, **kwargs):
        pass

    def forward_value(self, seq: torch.LongTensor, attention_mask: torch.LongTensor, prompt_length: int, labels: List[int], *args, **kwargs) -> Dict:
        if prompt_length > 0:
            seq = seq[:, prompt_length:]
        decoded_outputs = self.tokenizer.batch_decode(seq, skip_special_tokens=True)

        regrex = "A|B|C|D|E"
        preds = [re.findall(regrex, text) for text in decoded_outputs]

        rewards = []
        for pred, label in zip(preds, labels):
            if len(pred) == 0:
                rewards.append(0)
            else:
                rewards.append(int(self.option2int[pred[-1]] == label))

        rewards = torch.tensor(rewards, dtype=torch.bfloat16, device=seq.device)
        return {
            "values": rewards,
            "chosen_end_scores": rewards,
        }
