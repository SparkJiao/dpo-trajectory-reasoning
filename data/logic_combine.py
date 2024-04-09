import copy
import json
import os.path
from typing import List, Dict, Callable, Union

import omegaconf
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data.logiqav2 import _format_option_list
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def input_extract_aligner(original_data: List[Dict], input_index_field: str = "id", mode: str = "multi", correct_mode: str = "all"):
    if correct_mode == "all":
        flags = [0, 1]
    elif correct_mode == "correct":
        flags = [1]
    elif correct_mode == "wrong":
        flags = [0]
    else:
        raise ValueError(f"Unknown correct_mode: {correct_mode}")

    def func(response_data: List[Dict]):
        id2input_data = {item[input_index_field]: item for item in original_data}

        outputs = []
        for item in response_data:
            item_id = item["id"]
            response = item["response"]
            pred = item["pred"]
            if isinstance(response, str):
                response = [response]
                assert isinstance(pred, str), f"pred is not str: {pred}"
                pred = [pred]

            input_data = id2input_data[item_id]
            tmp = []
            for i, (resp, pred) in enumerate(zip(response, pred)):
                new_item = copy.deepcopy(input_data)
                new_item["response"] = resp
                new_item[input_index_field] = f"{item_id}_{i}"

                if pred and ord(pred.strip()) - ord("A") == new_item["label"]:
                    v = 1
                else:
                    v = 0
                if v in flags:
                    tmp.append(new_item)

            if mode == "multi":
                outputs.extend(tmp)
            elif mode == "single":
                if len(tmp) > 0:
                    outputs.append(tmp[0])
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return outputs

    return func


def field_extract_aligner(input_index_field: str, extract_index_field: str, extract_fields: List[str], extra_file: str):
    extra_data = json.load(open(extra_file))
    id2extra_data = {item[extract_index_field]: item for item in extra_data}

    def func(data: List[Dict]):
        outputs = []
        for item in data:
            item_id = item[input_index_field]
            if item_id not in id2extra_data:
                continue
            extra_item = id2extra_data[item_id]
            for field in extract_fields:
                item[field] = extra_item[field]
            outputs.append(item)

        logger.info(f"Extracted {len(outputs)} items from {extra_file}")

        return outputs

    return func


def flat_aligner(input_index_field: str, extract_field: Union[str, List[str]], mode: str = "single"):
    if isinstance(extract_field, str):
        extract_field = [extract_field]

    def func(data: List[Dict]):
        outputs = []
        for item in data:
            item_id = item[input_index_field]

            num = len(item[extract_field[0]])
            for _field in extract_field[1:]:
                assert len(item[_field]) == num, f"Length not match: {item[_field]}"

            for i in range(num):
                new_item = copy.deepcopy(item)
                for _field in extract_field:
                    new_item[_field] = item[_field][i]
                new_item[input_index_field] = f"{item_id}_{i}"
                outputs.append(new_item)
                if mode == "single":
                    break
            # for x_i, x in enumerate(item[extract_field]):
            #     new_item = copy.deepcopy(item)
            #     new_item[extract_field] = x
            #     new_item[input_index_field] = f"{item_id}_{x_i}"
            #     outputs.append(new_item)
            #     if mode == "single":  # The first item is that only contains the prefix and the first wrong step.
            #         break
        return outputs

    return func


def option_flatten_aligner():
    def func(data: List[Dict]):
        for sample in data:
            sample["option_list"] = _format_option_list(sample["options"], ["A", "B", "C", "D"])
        return data

    return func


def empty_aligner(data: List[Dict]):
    return data


def json_read_fn(file_path: str):
    return json.load(open(file_path))


def add_id_aligner(id_field: str = "id"):
    def func(data: List[Dict]):
        for i, item in enumerate(data):
            item[id_field] = i
        return data

    return func


class ResponseAlignDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 template: str,
                 aligner: Callable = empty_aligner,
                 instruction: str = "",
                 few_shot_prompt: str = "",
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None,
                 split_size: int = -1,
                 split_id: int = 0,
                 index_field: str = "id",
                 max_data_num: int = -1,
                 read_fn: Callable = json_read_fn,
                 **kwargs):
        self.tokenizer = tokenizer
        self.template = template
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor
        self.flush_file = flush_file
        self.split_size = split_size
        self.split_id = split_id
        self.index_field = index_field
        self.max_data_num = max_data_num

        data = read_fn(file_path)
        self.data: List[Dict] = aligner(data)

        for item in self.data:
            if self.instruction:
                item["instruction"] = self.instruction
            if self.few_shot_prompt:
                item["few_shot_prompt"] = self.few_shot_prompt

        flushed_data = set()
        if flush_file is not None and os.path.exists(flush_file):
            tmp = open(flush_file, "r").readlines()
            for line in tmp:
                item = json.loads(line)
                if "response" in item and item["response"]:
                    flushed_data.add(item["id"])
            logger.info(f"Loaded flushed data: {len(flushed_data)} from {flush_file}")

        if split_size > 0:
            batch_size = (len(self.data) + split_size - 1) // split_size
            self.data = self.data[split_id * batch_size: (split_id + 1) * batch_size]

        self.data = [item for item in self.data if item[self.index_field] not in flushed_data and str(item[self.index_field]) not in flushed_data]

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.data))
        return len(self.data)

    def api_getitem(self, index):
        item = self.data[index]
        text = self.template.format(**item)
        item["text"] = text
        return {
            "text": text,
            "meta_data": item,
        }

    def service_getitem(self, index):
        inputs = self.api_getitem(index)
        response = self.service_processor(inputs["text"])
        inputs["response"] = response
        return inputs

    def __getitem__(self, idx):
        if self.api_based:
            return self.api_getitem(idx)
        if self.service_based:
            return self.service_getitem(idx)
        item = self.data[idx]
        text = self.template.format(**item)
        item["text"] = text
        return {
            "text": text,
            "meta_data": item,
        }


class PromptResponseDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 prompt_template: str,
                 response_template: str,
                 aligner: Callable = empty_aligner,
                 instruction: str = "",
                 few_shot_prompt: str = "",
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None,
                 split_size: int = -1,
                 split_id: int = 0,
                 index_field: str = "id",
                 max_data_num: int = -1,
                 read_fn: Callable = json_read_fn,
                 kv_mapping: Dict[str, str] = None,
                 **kwargs):
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor
        self.flush_file = flush_file
        self.split_size = split_size
        self.split_id = split_id
        self.index_field = index_field
        self.max_data_num = max_data_num
        self.kv_mapping = kv_mapping

        data = read_fn(file_path)
        self.data: List[Dict] = aligner(data)

        for item in self.data:
            if self.instruction:
                item["instruction"] = self.instruction
            if self.few_shot_prompt:
                item["few_shot_prompt"] = self.few_shot_prompt

        flushed_data = set()
        if flush_file is not None and os.path.exists(flush_file):
            tmp = open(flush_file, "r").readlines()
            for line in tmp:
                item = json.loads(line)
                if "response" in item and item["response"].strip() != "":
                    flushed_data.add(item["id"])
            logger.info(f"Loaded flushed data: {len(flushed_data)} from {flush_file}")

        self.data = [item for item in self.data if item[self.index_field] not in flushed_data]

        if split_size > 0:
            batch_size = (len(self.data) + split_size - 1) // split_size
            self.data = self.data[split_id * batch_size: (split_id + 1) * batch_size]

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.data))
        return len(self.data)

    def api_getitem(self, index):
        raise NotImplementedError

    def service_getitem(self, index):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.api_based:
            return self.api_getitem(idx)
        if self.service_based:
            return self.service_getitem(idx)
        item = self.data[idx]
        prompt = self.prompt_template.format(**item)
        response = self.response_template.format(**item)
        text = prompt + response
        item["text"] = text
        item["prompt"] = prompt

        if not self.kv_mapping:
            return {
                "text": text,
                "meta_data": item,
            }

        res = {v: item[k] for k, v in self.kv_mapping.items()}
        res["meta_data"] = item
        return res


class MultiMappingDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 template: Dict[str, str],
                 aligner: Callable = empty_aligner,
                 instruction: str = "",
                 few_shot_prompt: str = "",
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None,
                 split_size: int = -1,
                 split_id: int = 0,
                 index_field: str = "id",
                 max_data_num: int = -1,
                 read_fn: Callable = json_read_fn,
                 kv_mapping: Dict[str, str] = None,
                 **kwargs):
        self.tokenizer = tokenizer
        self.template = template
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor
        self.flush_file = flush_file
        self.split_size = split_size
        self.split_id = split_id
        self.index_field = index_field
        self.max_data_num = max_data_num
        self.kv_mapping = kv_mapping

        data = read_fn(file_path)
        self.data: List[Dict] = aligner(data)

        for item in self.data:
            if self.instruction:
                item["instruction"] = self.instruction
            if self.few_shot_prompt:
                item["few_shot_prompt"] = self.few_shot_prompt

        flushed_data = set()
        if flush_file is not None and os.path.exists(flush_file):
            tmp = open(flush_file, "r").readlines()
            for line in tmp:
                item = json.loads(line)
                if "response" in item and item["response"]:
                    flushed_data.add(item["id"])
            logger.info(f"Loaded flushed data: {len(flushed_data)} from {flush_file}")

        self.data = [item for item in self.data if item[self.index_field] not in flushed_data]

        if split_size > 0:
            batch_size = (len(self.data) + split_size - 1) // split_size
            self.data = self.data[split_id * batch_size: (split_id + 1) * batch_size]

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.data))
        return len(self.data)

    def api_getitem(self, index):
        raise NotImplementedError

    def service_getitem(self, index):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.api_based:
            return self.api_getitem(idx)
        if self.service_based:
            return self.service_getitem(idx)
        item = self.data[idx]

        inputs = {}
        for k, v in self.template.items():
            item[k] = v.format(**item)
            inputs[k] = item[k]
        inputs["meta_data"] = item

        if not self.kv_mapping:
            return inputs

        res = {v: item[k] for k, v in self.kv_mapping.items()}
        res["meta_data"] = item
        return res
