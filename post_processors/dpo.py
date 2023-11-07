import json
import json
import os
from typing import Dict, Any

import numpy as np
import torch
from torch import distributed as dist

from post_processors.dist_mixin import DistGatherMixin


class DPOEvalPostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.chosen_rewards = []
        self.rejected_rewards = []
        self.losses = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        chosen = meta_data["chosen"]
        rejected = meta_data["reject"]

        chosen_rewards = batch_model_outputs["chosen_reward"].item()
        rejected_rewards = batch_model_outputs["rejected_reward"].item()
        loss = batch_model_outputs["loss"].item()

        if ddp:
            obj = [inputs, chosen, rejected, index, chosen_rewards, rejected_rewards, loss]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                chosen = []
                rejected = []
                index = []
                chosen_rewards = []
                rejected_rewards = []
                loss = []
                for item in gather_res:
                    inputs.extend(item[0])
                    chosen.extend(item[1])
                    rejected.extend(item[2])
                    index.extend(item[3])
                    chosen_rewards.append(item[4])
                    rejected_rewards.append(item[5])
                    loss.append(item[6])

        self.predictions.extend([{
            "input": input,
            "chosen": chosen,
            "rejected": rejected,
            "index": index,
        } for input, chosen, rejected, index in zip(inputs, chosen, rejected, index)])
        self.chosen_rewards.append(chosen_rewards)
        self.rejected_rewards.append(rejected_rewards)
        self.losses.append(loss)

    def get_results(self, output_dir: str):
        # output_file = os.path.join(output_dir, "eval_predictions.npy")
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        avg_loss = np.mean(self.losses).item()
        avg_chosen_reward = np.mean(self.chosen_rewards).item()
        avg_rejected_reward = np.mean(self.rejected_rewards).item()

        metrics = {
            "loss": avg_loss,
            "chosen_reward": avg_chosen_reward,
            "rejected_reward": avg_rejected_reward,
        }

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)
        json.dump(metrics, open(os.path.join(output_dir, "eval_metrics.json"), "w"), indent=2, ensure_ascii=False)

        return metrics, self.predictions
