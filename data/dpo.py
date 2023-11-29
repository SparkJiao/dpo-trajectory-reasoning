import collections
import json
import random
from typing import List, Dict, Tuple, Union, Any, Callable, Optional

import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class DPOPairReader:
    def __call__(self, file):
        data = json.load(open(file))
        return data


class DPOReaderAux(DPOPairReader):
    def __init__(self, extra_file):
        self.extra_file = extra_file

    def __call__(self, file):
        files = [file] + [self.extra_file]
        data = []
        for file in files:
            data += super().__call__(file)
        return data


class ComposeDatasetMixin(Dataset):
    def __init__(self, template: str = "", instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 ):
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def compose_input(self, item, response: str):
        _input = ""
        if self.instruction:
            _input += self.instruction + "\n\n"
        if self.few_shot_prompts:
            _input += self.few_shot_prompts + "\n\n"
        params = [item[k] for k in self.compose_keys]
        prompt = _input + self.template.format(*params)
        output = prompt + response
        return prompt, output


class DPOMergeDataset(ComposeDatasetMixin):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 reader=DPOPairReader(),
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        dpo_data = reader(file_path)
        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for pair_sample in self.id2dpo_item[item_id]:
                    if not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = pair_sample["chosen"]
                    reject = pair_sample["reject"]
                    # assert "is_full" in pair_sample, pair_sample  # Just for debug. Please comment this if you're sure the data is correct.
                    if "is_full" not in pair_sample or pair_sample["is_full"]:
                        chosen = chosen + tokenizer.eos_token
                        reject = reject + tokenizer.eos_token
                    item["chosen"] = chosen
                    item["reject"] = reject
                    item["index"] = item_id
                    data.append(item)

        logger.info(f"DPOMergeReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned[:10])}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item["chosen"]
        reject = item["reject"]
        if isinstance(chosen, list):
            chosen = random.choice(chosen)
        if isinstance(reject, list):
            reject = random.choice(reject)

        chosen_prompt, chosen_input = self.compose_input(item, chosen)
        reject_prompt, reject_input = self.compose_input(item, reject)
        assert chosen_prompt == reject_prompt, (chosen_prompt, reject_prompt)

        return {
            "prompt": chosen_prompt,
            "chosen": chosen_input,
            "reject": reject_input,
            "index": item["index"],
        }


class DPOSFTDataset(ComposeDatasetMixin):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 reader: Optional[Callable] = DPOPairReader(),
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)
        self.tokenizer = tokenizer

        dpo_data = reader(file_path)
        # Filter chosen data
        chosen_set = set()
        filtered_data = []
        for item in dpo_data:
            if item["chosen"] in chosen_set:
                continue
            chosen_set.add(item["chosen"])
            filtered_data.append(item)
        dpo_data = filtered_data

        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for pair_sample in self.id2dpo_item[item_id]:
                    if not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = pair_sample["chosen"]
                    # assert "is_full" in pair_sample, pair_sample  # Just for debug. Please comment this if you're sure the data is correct.
                    if "is_full" not in pair_sample or pair_sample["is_full"]:
                        chosen = chosen + tokenizer.eos_token

                    item["chosen"] = chosen
                    item["index"] = item_id
                    data.append(item)

        logger.info(f"DPOMergeReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned[:10])}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item["chosen"]
        if isinstance(chosen, list):
            chosen = random.choice(chosen)

        chosen_prompt, chosen_input = self.compose_input(item, chosen)

        return {
            "prompt": chosen_prompt,
            "chosen": chosen_input,
            "index": item["index"],
        }


class ReActFormat:
    def __call__(self, item):
        end_format = "Finish["
        lines = item["chosen"].split("\n")
        cnt = len([line for line in lines if end_format in line])
        if "is_full" not in item or item["is_full"]:
            if cnt > 1:
                return False
        else:
            if cnt > 0:
                return False
        return True


class DPOMergeBalanceDataset(ComposeDatasetMixin):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 reader: Optional[Callable] = DPOPairReader(),
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 balance_ratio: float = 1.0,
                 format_filter: Optional[Callable] = None,
                 ):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)
        self.tokenizer = tokenizer

        dpo_data = reader(file_path)
        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        full_data = []
        part_data = []
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for pair_sample in self.id2dpo_item[item_id]:
                    if not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = pair_sample["chosen"]
                    reject = pair_sample["reject"]
                    # assert "is_full" in pair_sample, pair_sample  # Just for debug. Please comment this if you're sure the data is correct.
                    if "is_full" not in pair_sample or pair_sample["is_full"]:
                        chosen = chosen + tokenizer.eos_token
                        reject = reject + tokenizer.eos_token

                    item["chosen"] = chosen
                    item["reject"] = reject
                    item["index"] = item_id

                    if "is_full" not in pair_sample or pair_sample["is_full"]:
                        full_data.append(item)
                    else:
                        part_data.append(item)

        logger.info("DPOMergeReader: {} / {} / {}".format(len(full_data), len(part_data), len(original_data)))
        self.full_data: List[Dict[str, Any]] = full_data
        self.part_data: List[Dict[str, Any]] = part_data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys
        self.balance_ratio = balance_ratio

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned[:10])}")

    def __len__(self):
        return len(self.full_data) + int(len(self.part_data) * self.balance_ratio)

    def __getitem__(self, index):
        # item = self.data[index]
        if index < len(self.full_data):
            item = self.full_data[index]
        else:
            # Randomly select sample from partial data
            item = random.choice(self.part_data)

        chosen = item["chosen"]
        reject = item["reject"]
        if isinstance(chosen, list):
            chosen = random.choice(chosen)
        if isinstance(reject, list):
            reject = random.choice(reject)

        chosen_prompt, chosen_input = self.compose_input(item, chosen)
        reject_prompt, reject_input = self.compose_input(item, reject)
        assert chosen_prompt == reject_prompt, (chosen_prompt, reject_prompt)

        return {
            "prompt": chosen_prompt,
            "chosen": chosen_input,
            "reject": reject_input,
            "index": item["index"],
        }


class DPOMergeParallelDataset(DPOMergeBalanceDataset):
    """
    This dataset is to ensure that the ration between full examples and partial examples is exactly 1:1.
    """

    def __len__(self):
        assert self.balance_ratio == 1
        return len(self.full_data)

    def process_item(self, item):
        chosen = item["chosen"]
        reject = item["reject"]
        if isinstance(chosen, list):
            chosen = random.choice(chosen)
        if isinstance(reject, list):
            reject = random.choice(reject)

        chosen_prompt, chosen_input = self.compose_input(item, chosen)
        reject_prompt, reject_input = self.compose_input(item, reject)
        assert chosen_prompt == reject_prompt, (chosen_prompt, reject_prompt)

        return {
            "prompt": chosen_prompt,
            "chosen": chosen_input,
            "reject": reject_input,
            "index": item["index"],
        }

    def __getitem__(self, index):
        full_item = self.full_data[index]
        part_item = random.choice(self.part_data)
        full_item = self.process_item(full_item)
        part_item = self.process_item(part_item)
        return {
            "full_input": full_item,
            "part_input": part_item,
        }


class DPOCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        reject = [item["reject"] for item in batch]
        indices = [item["index"] for item in batch]

        text_prompts = prompt + prompt
        text_inputs = chosen + reject

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "chosen": chosen,
            "reject": reject,
        }
        return encoded_inputs


class DPOParallelCollator(DPOCollator):
    def __call__(self, batch):
        full_batch = [item["full_input"] for item in batch]
        part_batch = [item["part_input"] for item in batch]
        combined_batch = full_batch + part_batch
        inputs = super().__call__(combined_batch)

        return inputs


class DPODataSFTCollator:
    """
    Note that when you are using the DPO pair dataset, you may overlook the oversampling of chosen samples.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        indices = [item["index"] for item in batch]

        text_prompts = prompt
        text_inputs = chosen

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "chosen": chosen,
            "response": chosen,
        }
        return encoded_inputs


class PairwiseRewardDataset(ComposeDatasetMixin):
    """
    This dataset reuse the logic from `DPOMergeDataset` and `DPOCollator`. The only difference is that we add `eos` token to all examples
    to ensure that the rewarding process is consistent across all examples.
    """

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 reader=DPOPairReader(),
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        dpo_data = reader(file_path)
        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for pair_sample in self.id2dpo_item[item_id]:
                    if not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = pair_sample["chosen"]
                    reject = pair_sample["reject"]
                    chosen = chosen + tokenizer.eos_token
                    reject = reject + tokenizer.eos_token

                    item["chosen"] = chosen
                    item["reject"] = reject
                    item["index"] = item_id
                    data.append(item)

        logger.info(f"DPOMergeReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned[:10])}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        chosen = item["chosen"]
        reject = item["reject"]
        if isinstance(chosen, list):
            chosen = random.choice(chosen)
        if isinstance(reject, list):
            reject = random.choice(reject)

        chosen_prompt, chosen_input = self.compose_input(item, chosen)
        reject_prompt, reject_input = self.compose_input(item, reject)
        assert chosen_prompt == reject_prompt, (chosen_prompt, reject_prompt)

        return {
            "prompt": chosen_prompt,
            "chosen": chosen_input,
            "reject": reject_input,
            "index": item["index"],
        }
