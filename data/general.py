import copy
import json
import os.path
from typing import List, Dict, Tuple, Union, Any, Callable, Optional
from glob import glob
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
import collections
import torch
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from general_util.logger import get_child_logger
from data.logiqav2 import LogicQAReader
from data.dpo import ComposeDatasetMixin
import re

logger = get_child_logger(__name__)

templates = [
    "Context:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\nReasoning process:\n{}\n\nModified reasoning process with flaw:\n",
    "Context:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\nReasoning process:\nThought 1: {}\n\nModified reasoning process with flaw:\n",
    # "Context:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\nReasoning process:\nThought 1: {}\n\nModified reasoning process with flaw:\n [/INST]",
]

MODIFICATION_WORSE_V1 = ("Here is a reasoning problem and its reasoning process. The reasoning process may not be completed. Modify the reasoning process "
                         "to make it a incorrect one.")
MODIFICATION_WORSE_V2 = ("Here is a reasoning problem and its reasoning process. Modify the reasoning process to make it an incorrect one. Note that the "
                         "reasoning process may not be completed, but you can still modify it to make it an incorrect one.")
MODIFICATION_WORSE_V2_Mistral = ("[INST] Here is a reasoning problem and its reasoning process. Modify the reasoning process to make it an incorrect one. "
                                 "Note that the reasoning process may not be completed, but you can still modify it to make it an incorrect one. [/INST]")
MODIFICATION_WORSE_V3 = "Here is a reasoning problem and its reasoning process. Modify the reasoning process to make it an incorrect one."
MODIFICATION_WORSE_V4_Mistral = ("[INST] Here is a reasoning problem and its reasoning process. Modify the reasoning process to make it an incorrect one. "
                                 "Note that the reasoning process may not be completed, but you can still modify it to make it an incorrect one. "
                                 "Do not explain why this modification is incorrect. [/INST]")

prompts = {
    "modify_worse_v1": MODIFICATION_WORSE_V1,
    "modify_worse_v2": MODIFICATION_WORSE_V2,
    "modify_worse_v2_mistral": MODIFICATION_WORSE_V2_Mistral,
    "modify_worse_v3": MODIFICATION_WORSE_V3,
    "modify_worse_v4_mistral": MODIFICATION_WORSE_V4_Mistral,
}


def get_prompt(prompt_name: str) -> str:
    return prompts[prompt_name]


def get_template(template_id: int) -> str:
    return templates[template_id]


def process_response(response: str):
    lines = response.split("\n")
    lines = list(filter(lambda x: x[1].startswith("Thought ") or x[1].startswith("Action ") or x[1].startswith("Observation "), enumerate(lines)))
    return lines


def clean_react_response(response: str):
    if "Context:\n" in response:
        response = response.split("Context:\n")[0]

    lines = response.split("\n")
    finish_lines = [lines for line in lines if "Finish[The answer is" in line and line.startswith("Action ")]
    if len(finish_lines) != 1:
        return None

    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "Finish[The answer is" in line and line.startswith("Action "):
            break
    assert "Finish[The answer is" in new_lines[-1] and new_lines[-1].startswith("Action "), new_lines
    response = "\n".join(new_lines)
    response = response.strip()
    return response


def parse_leaf_node_value(response: str, label: int):
    groups = response.split("Finish")
    if len(groups) < 2:
        # print(f"Warning: Not a valid response: {response}")
        return 0
    response = groups[1]
    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        return 0
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            return 0


def extract_react_correct_response(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_data = []
    for item in data:
        response = item["response"]
        preds = item["pred"]
        if isinstance(response, str):
            response = [response]
            preds = [preds]
        new_responses = []
        new_preds = []
        for res, pred in zip(response, preds):
            res = clean_react_response(res)
            if res is None:
                continue
            new_responses.append(res)
            new_preds.append(pred)
        if len(new_responses) == 0:
            continue
        item["response"] = new_responses
        item["pred"] = new_preds
        cleaned_data.append(item)

    logger.info(f"ReAct Clean: {len(data)} -> {len(cleaned_data)}")

    correct = []
    for item in cleaned_data:
        for resp_id, resp in enumerate(item["response"]):
            v = parse_leaf_node_value(resp, item["label"])
            if v == 1:
                correct.append({
                    "response": resp,
                    "id": f"{item['id']}-{resp_id}",
                })

    logger.info(f"ReAct Correct: {len(correct)}")
    return correct


class PartialTrajAttemptsReader:
    def __init__(self, partial_traj_file: str):
        self.partial_traj_file = partial_traj_file

    def __call__(self, attempt_response_file):
        if os.path.exists(attempt_response_file):
            files = [attempt_response_file]
        else:
            files = glob(attempt_response_file)

        state_id2values = collections.defaultdict(dict)
        for file in files:
            data = json.load(open(file, "r"))

            for item in data:
                idx = item["id"]
                idx, state_id = idx.split("_")
                v = 0
                flag = True
                for res, p in zip(item["response"], item["pred"]):
                    if res == "":
                        flag = False
                        break
                    if p == "":
                        continue
                    if ord(p.strip()) - ord("A") == item["label"]:
                        v += 1
                if not flag:
                    continue
                state_id2values[int(idx)][int(state_id)] = (v, item["text"].split("Thought 1:")[1].strip())

        logger.info(f"Loaded {len(state_id2values)} state values from {len(files)} files.")

        if os.path.exists(self.partial_traj_file):
            inter_state_files = [self.partial_traj_file]
        else:
            inter_state_files = glob(self.partial_traj_file)
        inter_states = []
        for state_file in inter_state_files:
            inter_states.extend(json.load(open(state_file, "r")))

        logger.info(f"Loaded {len(inter_states)} inter states from {len(inter_state_files)} files.")

        jumped = 0
        outputs = []
        for item in tqdm(inter_states, total=len(inter_states)):
            idx = item["id"]
            if idx not in state_id2values:
                jumped += 1
                continue

            for s_id, s in enumerate(item["inter_states"]):
                if s_id not in state_id2values[idx]:
                    continue
                s["value"] = int(state_id2values[idx][s_id][0])
                assert state_id2values[idx][s_id][1] in s["state"], (state_id2values[idx][s_id][1], s["state"])

            outputs.append(item)

        logger.info(f"Jumped {jumped} inter states.")

        return outputs


class ReActResponseMergeReader:
    def __init__(self, response_file: str, original_reader: LogicQAReader = LogicQAReader()):
        self.original_reader = original_reader
        self.responses = extract_react_correct_response(json.load(open(response_file, "r")))

    def __call__(self, file):
        original_data = self.original_reader(file)
        id2original_data = {idx: item for idx, item in enumerate(original_data)}

        outputs = []
        for item in self.responses:
            item_id, resp_id = item["id"].split("-")
            item_id = int(item_id)
            resp_id = int(resp_id)
            original_item = id2original_data[item_id]
            new_item = copy.deepcopy(original_item)
            new_item["response"] = item["response"]
            new_item["id"] = f"{item_id}_{resp_id}"
            outputs.append(new_item)
        return outputs


class WorsenInterStateMergeReader:
    def __init__(self, response_file, original_reader: LogicQAReader = LogicQAReader()):
        self.original_reader = original_reader
        self.responses = json.load(open(response_file, "r"))

    def __call__(self, file):
        original_data = self.original_reader(file)
        id2original_data = {idx: item for idx, item in enumerate(original_data)}

        outputs = []
        for item in self.responses:
            item_id, inter_state_id = item["id"].split("_")
            item_id = int(item_id)
            assert item_id in id2original_data, item_id
            original_item = id2original_data[item_id]
            new_item = copy.deepcopy(original_item)
            new_item["response"] = item["response"].strip()
            new_item["id"] = f"{item_id}_{inter_state_id}"
            outputs.append(new_item)
        return outputs


class Attempt2ValueRewardModelingDataset(ComposeDatasetMixin):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str, reader: Callable,
                 instruction: str = "", few_shot_prompts: str = "", max_value: int = 3,
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False, ):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        inter_states = reader(file_path)
        self.id2inter_states = collections.defaultdict(list)
        for item in inter_states:
            self.id2inter_states[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2inter_states:
                for inter_item in self.id2inter_states[item_id]:
                    if format_filter is not None and not format_filter(inter_item):
                        abandoned.append(inter_item)
                        continue

                    for state in inter_item["inter_states"]:
                        if "value" in state:
                            response = state["state"]
                            value = state["value"]
                            new_item = copy.deepcopy(item)
                            new_item["response"] = response
                            new_item["value"] = value
                            new_item["index"] = item_id if not re_index else len(data)
                            data.append(new_item)
                    for full_response in inter_item["response"]:
                        value = parse_leaf_node_value(full_response, item["label"]) * max_value
                        new_item = copy.deepcopy(item)
                        new_item["response"] = full_response
                        new_item["value"] = value
                        new_item["index"] = item_id if not re_index else len(data)
                        data.append(new_item)

        logger.info(f"Attempt2ValueRewardModelingDataset: {len(data)} / {len(original_data)}")
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
        prompt, _input = self.compose_input(item, item["response"])

        return {
            "prompt": prompt,
            "response": item["response"],
            "input": _input,
            "value": item["value"],
            "index": item["index"],
        }


class CompleteTrajRewardModelingDataset(ComposeDatasetMixin):
    """
    This dataset only loads the complete trajectory.
    """

    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str, reader: Callable,
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False, ):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        if os.path.exists(file_path):
            files = [file_path]
        else:
            files = glob(file_path)
        logger.info(f"Loading {len(files)} files from {file_path}.")
        responses = []
        for file in files:
            responses.extend(reader(file))
        self.id2response = collections.defaultdict(list)
        for item in responses:
            self.id2response[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2response:
                for response_item in self.id2response[item_id]:
                    if format_filter is not None and not format_filter(response_item):
                        abandoned.append(response_item)
                        continue

                    for full_response in response_item["response"]:
                        new_item = copy.deepcopy(item)
                        new_item["response"] = full_response
                        new_item["index"] = item_id if not re_index else len(data)
                        data.append(new_item)

        logger.info(f"Attempt2ValueRewardModelingDataset: {len(data)} / {len(original_data)}")
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
        prompt, _input = self.compose_input(item, item["response"])

        return {
            "prompt": prompt,
            "response": item["response"],
            "input": _input,
            "index": item["index"],
        }


class Attempt2ValueCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        inputs = [item["input"] for item in batch]
        indices = [item["index"] for item in batch]
        values = [item["value"] for item in batch]

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
        encoded_inputs["values"] = torch.tensor(values, dtype=torch.long)
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "input": inputs,
            "response": [item["response"] for item in batch],
            "label": values,
        }
        return encoded_inputs


def extract_react_ending_positions(tokenizer: PreTrainedTokenizer, response: str, max_seq_length: int):
    steps = process_response(response)
    raw_lines = response.split("\n")
    endings = []
    resp_start = False
    for step_id, step in steps:
        if not resp_start:
            if step.startswith("Thought 1:"):
                resp_start = True
            else:
                continue
        partial_traj = "\n".join(raw_lines[:(step_id + 1)])
        input_ids = tokenizer(partial_traj, truncation=True, max_length=max_seq_length)["input_ids"]
        endings.append(len(input_ids) - 1)

    assert resp_start, response
    assert len(endings) > 0, (response, steps)
    assert len(endings) == len(process_response(response[response.find("Thought 1:"):])), (response, steps)
    return endings


class CompleteTrajStepRewardCollator:
    """
    This is used on par with `CompleteTrajRewardModelingDataset`.

    This dataset splits the trajectory into steps, and takes notes of the ending positions of each step.
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

        endings = []
        padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
        for b, item in enumerate(batch):
            ending = extract_react_ending_positions(self.tokenizer, item["input"], self.max_seq_length)
            if self.tokenizer.padding_side == "left":
                ending = [e + padding_len[b].item() for e in ending]
            endings.append(ending)
            # tmp = process_response("Thought 1:" + item["input"].split("Thought 1:")[1])
            # tmp2 = process_response("Thought 1: " + item["response"])
            # assert len(ending) == len(tmp), (item["input"], ending)
            # print("A", len(ending))
            # print("B", len(tmp))
            # print("C", len(tmp2))

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "input": inputs,
            "response": [item["response"] for item in batch],
            "ending": endings,
        }
        return encoded_inputs
