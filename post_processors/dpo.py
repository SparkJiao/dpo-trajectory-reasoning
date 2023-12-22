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
        json.dump(metrics, open(output_file.replace(".json", ".metrics.json"), "w"), indent=2, ensure_ascii=False)

        return metrics, self.predictions


class DPORewardPostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []
        self.losses = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        chosen = meta_data["chosen"]
        rejected = meta_data["reject"]

        chosen_rewards = batch_model_outputs["batch_chosen_reward"].tolist()
        rejected_rewards = batch_model_outputs["batch_rejected_reward"].tolist()
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
                    chosen_rewards.extend(item[4])
                    rejected_rewards.extend(item[5])
                    loss.append(item[6])

        self.predictions.extend([{
            "input": prompt,
            "chosen": ch,
            "rejected": rej,
            "index": i,
            "chosen_reward": chosen_r,
            "rejected_reward": rejected_r,
        } for prompt, ch, rej, chosen_r, rejected_r, i in zip(inputs, chosen, rejected, chosen_rewards, rejected_rewards, index)])
        self.losses.append(loss)

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        acc = np.mean([x["chosen_reward"] > x["rejected_reward"] for x in self.predictions]).item()

        metrics = {
            "acc": acc,
        }

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)
        json.dump(metrics, open(output_file.replace(".json", ".metrics.json"), "w"), indent=2, ensure_ascii=False)

        return metrics, self.predictions


class ResponseClsPostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        responses = meta_data["response"]
        labels = meta_data["label"]

        logits = batch_model_outputs["logits"].tolist()

        if ddp:
            obj = [inputs, index, responses, logits, labels]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                index = []
                responses = []
                logits = []
                labels = []
                for item in gather_res:
                    inputs.extend(item[0])
                    index.extend(item[1])
                    responses.extend(item[2])
                    logits.extend(item[3])
                    labels.extend(item[4])

        self.predictions.extend([{
            "input": prompt,
            "index": i,
            "response": resp,
            "logits": logit,
            "label": label,
        } for prompt, i, resp, logit, label in zip(inputs, index, responses, logits, labels)])

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        pred = [x["logits"] for x in self.predictions]
        pred = np.argmax(pred, axis=1).tolist()
        labels = [x["label"] for x in self.predictions]
        acc = np.mean([x == y for x, y in zip(pred, labels)]).item()

        metrics = {
            "acc": acc,
        }

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)
        json.dump(metrics, open(output_file.replace(".json", ".metrics.json"), "w"), indent=2, ensure_ascii=False)

        return metrics, self.predictions


class ResponseProcessRewardPostProcessor(DistGatherMixin):
    def __init__(self, reduction: str = "product"):
        """
        :param reduction: "product|min"
        """
        super().__init__()
        self.predictions = []
        self.reduction = reduction

    @staticmethod
    def logit2prob(logits):
        probs = torch.softmax(logits, dim=-1)
        probs = probs[:, 2] + probs[:, 3]
        return probs

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        responses = meta_data["response"]
        ending_positions = meta_data["ending"]

        logits = batch_model_outputs["logits"].tolist()

        ending_logits = []
        assert len(ending_positions) == len(logits)
        for endings, seq_logits in zip(ending_positions, logits):
            try:
                ending_logits.append([seq_logits[e] for e in endings])
            except IndexError:
                print(endings)
                print(len(seq_logits))
                raise

        if ddp:
            obj = [inputs, index, responses, ending_logits]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                index = []
                responses = []
                ending_logits = []
                for item in gather_res:
                    inputs.extend(item[0])
                    index.extend(item[1])
                    responses.extend(item[2])
                    ending_logits.extend(item[3])

        self.predictions.extend([{
            "input": prompt,
            "index": i,
            "response": resp,
            "ending_logits": logits,
        } for prompt, i, resp, logits in zip(inputs, index, responses, ending_logits)])

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        self.predictions = sorted(self.predictions, key=lambda x: x["index"])

        for pred in self.predictions:
            logits = torch.tensor(pred["ending_logits"])
            probs = self.logit2prob(logits)
            if self.reduction == "product":
                pred["reward"] = probs.prod().item()
            elif self.reduction == "min":
                pred["reward"] = probs.min().item()
            else:
                raise ValueError(f"Unknown reduction: {self.reduction}")

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)

        return {}, self.predictions


class DPORewardSinglePostProcessor(DistGatherMixin):
    def __init__(self):
        super().__init__()
        self.predictions = []

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], ddp: bool = False):
        index = meta_data["index"]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        inputs = meta_data["prompt"]
        responses = meta_data["response"]

        rewards = batch_model_outputs["batch_chosen_reward"].tolist()

        if ddp:
            obj = [inputs, responses, index, rewards]
            gather_res = self.gather_object(obj)
            if dist.get_rank() == 0:
                inputs = []
                responses = []
                index = []
                rewards = []
                for item in gather_res:
                    inputs.extend(item[0])
                    responses.extend(item[1])
                    index.extend(item[2])
                    rewards.extend(item[3])

        self.predictions.extend([{
            "input": prompt,
            "response": resp,
            "index": i,
            "reward": r,
        } for prompt, resp, r, i in zip(inputs, responses, rewards, index)])

    def get_results(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if dist.is_initialized():
            dist.barrier()

        if dist.is_initialized():
            output_file = os.path.join(output_dir, f"eval_predictions_rank{dist.get_rank()}.json")
        else:
            output_file = os.path.join(output_dir, "eval_predictions.json")

        json.dump(self.predictions, open(output_file, "w"), indent=2, ensure_ascii=False)

        return {}, self.predictions
