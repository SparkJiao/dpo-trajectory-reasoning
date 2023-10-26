from dataclasses import dataclass

import torch


@dataclass
class RewardModelOutputs:
    chosen_end_scores: torch.Tensor = None


class RewardModelMixin:
    def forward_value(self, *args, **kwargs) -> RewardModelOutputs:
        raise NotImplementedError
