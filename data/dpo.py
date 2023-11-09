import collections
import json
import random
from typing import List, Dict, Tuple, Union, Any, Callable

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


class DPOMergeDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str,
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"), ):
        self.tokenizer = tokenizer

        reader = DPOPairReader()
        dpo_data = reader(file_path)
        self.id2dpo_item = collections.defaultdict(list)
        for item in dpo_data:
            self.id2dpo_item[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2dpo_item:
                for pair_sample in self.id2dpo_item[item_id]:
                    chosen = pair_sample["chosen"]
                    reject = pair_sample["reject"]
                    # assert "is_full" in pair_sample, pair_sample  # Just for debug. Please comment this if you're sure the data is correct.
                    if getattr(pair_sample, "is_full", True):
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

    def __len__(self):
        return len(self.data)

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


class DPODataSFTCollator:
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
