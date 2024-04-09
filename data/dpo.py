import collections
import copy
import json
import os.path
import random
from glob import glob
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

    def compose_input(self, item, response: str, prefix: str = ""):
        _input = ""
        if self.instruction:
            _input += self.instruction + "\n\n"
        if self.few_shot_prompts:
            _input += self.few_shot_prompts + "\n\n"
        params = [item[k] for k in self.compose_keys]
        prompt = _input + self.template.format(*params)
        output = prompt + response
        if prefix != "":
            assert response.find(prefix) == 0, (response, prefix)
            prompt = prompt + prefix

        return prompt, output

    @staticmethod
    def append_eos(item, response_key, eos_token):
        if f"{response_key}_full" in item:
            if item[f"{response_key}_full"]:
                return item[response_key] + eos_token
            else:
                return item[response_key]
        else:
            if "is_full" in item:
                if item["is_full"]:
                    return item[response_key] + eos_token
                else:
                    return item[response_key]
            else:
                return item[response_key] + eos_token


class DPOMergeDataset(ComposeDatasetMixin):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 reader=DPOPairReader(),
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 index_field: str = "index"):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer
        self.index_field = index_field

        dpo_data = reader(file_path)
        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        for i, item in enumerate(original_data):
            if self.index_field in item:
                item_id = item[self.index_field]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for pair_sample in self.id2dpo_item[item_id]:
                    if format_filter is not None and not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = self.append_eos(pair_sample, "chosen", tokenizer.eos_token)
                    reject = self.append_eos(pair_sample, "reject", tokenizer.eos_token)
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
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

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


class DPOMergePrefixDataset(ComposeDatasetMixin):
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
                    if format_filter is not None and not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = self.append_eos(pair_sample, "chosen", tokenizer.eos_token)
                    reject = self.append_eos(pair_sample, "reject", tokenizer.eos_token)
                    item["chosen"] = chosen
                    item["reject"] = reject
                    assert "chosen_prefix" in pair_sample, pair_sample.keys()
                    if "chosen_prefix" in pair_sample:
                        item["chosen_prefix"] = pair_sample["chosen_prefix"]
                    if "reject_prefix" in pair_sample:
                        item["reject_prefix"] = pair_sample["reject_prefix"]
                    item["index"] = item_id
                    data.append(item)

        logger.info(f"DPOMergeReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

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

        chosen_prompt, chosen_input = self.compose_input(item, chosen, prefix=item["chosen_prefix"])
        reject_prompt, reject_input = self.compose_input(item, reject, prefix=item["reject_prefix"])
        # assert chosen_prompt == reject_prompt, (chosen_prompt, reject_prompt)

        return {
            "chosen_prompt": chosen_prompt,
            "reject_prompt": reject_prompt,
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
                    if format_filter is not None and not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = pair_sample["chosen"]
                    # assert "is_full" in pair_sample, pair_sample  # Just for debug. Please comment this if you're sure the data is correct.
                    # if "is_full" not in pair_sample or pair_sample["is_full"]:
                    #     chosen = chosen + tokenizer.eos_token
                    chosen = self.append_eos(pair_sample, "chosen", tokenizer.eos_token)

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
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

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
        # TODO: Maybe we also need to implement `chosen_full` and `reject_full` options here.
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
                    # if "is_full" not in pair_sample or pair_sample["is_full"]:
                    #     chosen = chosen + tokenizer.eos_token
                    #     reject = reject + tokenizer.eos_token
                    chosen = self.append_eos(pair_sample, "chosen", tokenizer.eos_token)
                    reject = self.append_eos(pair_sample, "reject", tokenizer.eos_token)

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
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

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
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, padding: str = "longest"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding

    def __call__(self, batch):
        chosen = [item["chosen"] for item in batch]
        reject = [item["reject"] for item in batch]
        indices = [item["index"] for item in batch]
        text_inputs = chosen + reject

        text_prompts = []
        for item in batch:
            if "chosen_prompt" in item:
                text_prompts.append(item["chosen_prompt"])
            else:
                text_prompts.append(item["prompt"])
        for item in batch:
            if "reject_prompt" in item:
                text_prompts.append(item["reject_prompt"])
            else:
                text_prompts.append(item["prompt"])
        # prompt = [item["prompt"] for item in batch]
        # text_prompts = prompt + prompt

        encoded_prompts = self.tokenizer(text_prompts, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": text_prompts,
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
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False,
                 add_eos_token: bool = True):
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
                    if format_filter is not None and not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = pair_sample["chosen"]
                    reject = pair_sample["reject"]
                    if add_eos_token:
                        chosen = chosen + tokenizer.eos_token
                        reject = reject + tokenizer.eos_token

                    item["chosen"] = chosen
                    item["reject"] = reject
                    item["index"] = item_id if not re_index else i
                    data.append(item)

        logger.info(f"DPOMergeReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

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


class SingleRecordRewardDataset(ComposeDatasetMixin):
    """
    This dataset reuse the logic from `DPOMergeDataset` and `DPOCollator`. The only difference is that we add `eos` token to all examples
    to ensure that the rewarding process is consistent across all examples.

    This dataset will use `set` to process each example only once to reduce inference time. This is only used for inference.
    """

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 reader=DPOPairReader(),
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False, ):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer
        dpo_data = reader(file_path)

        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        response_set = set()
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for pair_sample in self.id2dpo_item[item_id]:
                    if format_filter is not None and not format_filter(pair_sample):
                        abandoned.append(pair_sample)
                        continue

                    chosen = pair_sample["chosen"]
                    reject = pair_sample["reject"]

                    if chosen not in response_set:
                        response_set.add(chosen)
                        # chosen = chosen + tokenizer.eos_token
                        # item["response"] = chosen
                        # item["index"] = item_id if not re_index else i
                        # data.append(item)
                        new_item = copy.deepcopy(item)
                        new_item["response"] = chosen
                        new_item["index"] = item_id if not re_index else i
                        data.append(new_item)

                    if reject not in response_set:
                        response_set.add(reject)
                        # reject = reject + tokenizer.eos_token
                        # item["response"] = reject
                        # item["index"] = item_id if not re_index else i
                        # data.append(item)
                        new_item = copy.deepcopy(item)
                        new_item["response"] = reject
                        new_item["index"] = item_id if not re_index else i
                        data.append(new_item)

        logger.info(f"Unique response number: {len(response_set)}")
        tmp = set([item["response"] for item in data])
        logger.info(f"Unique response number in data: {len(tmp)}")

        logger.info(f"DPOMergeReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        prompt, _input = self.compose_input(item, item["response"] + self.tokenizer.eos_token)

        return {
            "prompt": prompt,
            "response": item["response"],
            "input": _input,
            "index": item["index"],
        }


class InterStatesRewardingDataset(ComposeDatasetMixin):
    """
    This dataset read the original response file with intermediate states.

    This dataset is for evaluation only.
    """

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 reader=DPOPairReader(),
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False,
                 add_eos_token: bool = True, ):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        if os.path.exists(file_path):
            dpo_data = reader(file_path)
        else:
            dpo_data = []
            for file in glob(file_path):
                dpo_data += reader(file)
        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        response_set = set()
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for dpo_item in self.id2dpo_item[item_id]:
                    for state in dpo_item["inter_states"]:
                        response = state["state"]
                        if response not in response_set:
                            response_set.add(response)
                            new_item = copy.deepcopy(item)
                            new_item["response"] = response
                            new_item["index"] = item_id if not re_index else len(data)
                            data.append(new_item)

                    for full_response in dpo_item["response"]:
                        if full_response not in response_set:
                            response_set.add(full_response)
                            new_item = copy.deepcopy(item)
                            new_item["response"] = full_response
                            new_item["index"] = item_id if not re_index else len(data)
                            data.append(new_item)

        logger.info(f"Unique response number: {len(response_set)}")
        tmp = set([item["response"] for item in data])
        logger.info(f"Unique response number in data: {len(tmp)}")

        logger.info(f"DPOMergeReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys
        self.add_eos_token = add_eos_token

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if self.add_eos_token:
            prompt, _input = self.compose_input(item, item["response"] + self.tokenizer.eos_token)
        else:
            prompt, _input = self.compose_input(item, item["response"])

        return {
            "prompt": prompt,
            "response": item["response"],
            "input": _input,
            "index": item["index"],
        }


class SingleRecordRewardCollator:
    """
    This is used on par with `SingleRecordRewardDataset`.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        inputs = [item["input"] for item in batch]
        indices = [item["index"] for item in batch]

        text_prompts = prompt
        text_inputs = inputs

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
            "input": inputs,
            "response": [item["response"] for item in batch],
        }
        return encoded_inputs
