import collections
import copy
import json
import os.path
import random
import re
from glob import glob
from typing import List, Dict, Tuple, Union, Any, Callable, Optional

import omegaconf
import torch
from omegaconf.listconfig import ListConfig
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from data.dpo import ComposeDatasetMixin
from data.logiqav2 import LogicQAReader
from general_util.logger import get_child_logger

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


def process_response_v2(response: str):
    lines = response.split("\n")
    outputs = []
    for line_id, line in enumerate(lines):

        if line.startswith("Thought ") or line.startswith("Action ") or line.startswith("Observation "):
            outputs.append({
                "text": line,
                "type": "text",
                "line_id": line_id,
            })
        elif not line.strip():
            outputs.append({
                "text": line,
                "type": "space",
                "line_id": line_id,
            })
        else:
            outputs.append({
                "text": line,
                "type": "continue",
                "line_id": line_id,
            })

    compose_outputs = []
    for item in outputs:
        if item["type"] == "text":
            compose_outputs.append((item["line_id"], item["text"]))
        elif item["type"] == "space":
            if len(compose_outputs):
                tmp = compose_outputs[-1]
                new_line_text = "\n".join([tmp[1], item["text"]])
                compose_outputs[-1] = (tmp[0], new_line_text)
        else:
            if len(compose_outputs):
                tmp = compose_outputs[-1]
                new_line_text = "\n".join([tmp[1], item["text"]])
                compose_outputs[-1] = (item["line_id"], new_line_text)
            else:
                compose_outputs.append((item["line_id"], item["text"]))

    outputs = []
    types = []
    for item in compose_outputs:
        if item[1].startswith("Thought "):
            content = item[1][len("Thought "):]
            content = content.strip()
            if len(content) >= 5 or item[1].startswith(
                    "Thought 1:"):  # FIXED: Hack for LogiQA-v2 reward model evaluation, where the responses are not cleaned. @2024/01/18.
                outputs.append(item)
                types.append("Thought")
        elif item[1].startswith("Action "):
            content = item[1][len("Action "):]
            content = content.strip()
            if len(content) >= 5:
                outputs.append(item)
                types.append("Action")
        elif item[1].startswith("Observation "):
            content = item[1][len("Observation "):]
            content = content.strip()
            if len(content) >= 5:
                outputs.append(item)
                types.append("Observation")
        else:
            # logger.warning(f"Warning: Unknown line: {item[1]}")
            if len(item[1]) >= 5:
                outputs.append(item)
                types.append("Unknown")

    return outputs, types


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


def parse_leaf_node_value(response: str, label: int, logs: dict):
    groups = response.split("Finish")
    if len(groups) < 2:
        # print(f"Warning: Not a valid response: {response}")
        if "invalid" not in logs:
            logs["invalid"] = 0
        logs["invalid"] += 1
        return 0
    response = groups[1]
    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        if "missing" not in logs:
            logs["missing"] = 0
        logs["missing"] += 1
        return 0
    elif len(preds) > 1:
        if "multiple" not in logs:
            logs["multiple"] = 0
        logs["multiple"] += 1
        # print(f"Warning: Multiple answers: {response}")  # Fixed: Here is fixed.
        return 0
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            if "wrong" not in logs:
                logs["wrong"] = 0
            logs["wrong"] += 1
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


"""
Fixed: @2023-12-27

Add the ReAct parsing results function here to avoid the influence from previous unchanged post-processing code,
which did not filter out the solutions hacking the problem by predicting multiple answers.
"""


class PartialTrajAttemptsReader:
    def __init__(self, partial_traj_file: str):
        self.partial_traj_file = partial_traj_file

    def __call__(self, attempt_response_file):
        if os.path.exists(attempt_response_file):
            files = [attempt_response_file]
        else:
            files = glob(attempt_response_file)

        state_id2values = collections.defaultdict(dict)
        logs = {}
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
                    # if ord(p.strip()) - ord("A") == item["label"]:
                    #     v += 1
                    v += parse_leaf_node_value(res, item["label"], logs=logs)

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

        logger.info(f"Parsing inter states attempts logs: {logs}")
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


class PartialTrajAttemptsReaderV2:
    """
    In this version, we want to the model to be optimized on those responses ending with meaningful responses, instead of blank step, e.g., "Action 1: \n".
    Update @ 2024/01/08: Remove duplicates for more accurate estimation.
    """

    def __init__(self, partial_traj_file: str):
        self.partial_traj_file = partial_traj_file

    def __call__(self, attempt_response_file):
        if isinstance(attempt_response_file, omegaconf.ListConfig):
            files = list(attempt_response_file)
        elif os.path.exists(attempt_response_file):
            files = [attempt_response_file]
        else:
            files = glob(attempt_response_file)

        state_id2values = collections.defaultdict(dict)
        logs = {}
        for file in files:
            logger.info(f"Loading {file}")
            data = json.load(open(file, "r"))

            for item in data:
                idx = item["id"]
                idx, state_id = idx.split("_")
                v = 0
                flag = True
                resp_set = set([resp for resp in item["response"]])
                if len(resp_set) != len(item["response"]):
                    continue
                for res, p in zip(item["response"], item["pred"]):
                    if res == "":
                        flag = False
                        break
                    if p == "":
                        continue
                    v += parse_leaf_node_value(res, item["label"], logs=logs)

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

        logger.info(f"Parsing inter states attempts logs: {logs}")
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
                # Add filter here
                lines = s["state"].split("\n")
                lines = list(filter(lambda x: x.strip(), lines))
                last_state = lines[-1]
                if last_state.startswith("Thought "):
                    content = last_state[len("Thought "):]
                elif last_state.startswith("Action "):
                    content = last_state[len("Action "):]
                elif last_state.startswith("Observation "):
                    content = last_state[len("Observation "):]
                else:
                    content = last_state
                content = content.strip()
                if len(content) < 5:
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


class Value2LabelMapping:
    def __init__(self, name):
        if name == "greater_then_one":
            self.mapping = self.greater_then_one
        else:
            raise NotImplementedError

    @staticmethod
    def greater_then_one(value):
        return 1 if value > 0 else 0


class Attempt2ValueRewardModelingDataset(ComposeDatasetMixin):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str, reader: Callable, max_value: int,
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False,
                 value_mapping: Optional[Value2LabelMapping] = None):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        inter_states = reader(file_path)
        self.id2inter_states = collections.defaultdict(list)
        for item in inter_states:
            self.id2inter_states[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        logs = {}
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
                        value = parse_leaf_node_value(full_response, item["label"], logs) * max_value
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
        self.value_mapping = value_mapping

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        prompt, _input = self.compose_input(item, item["response"])

        if self.value_mapping is not None:
            value = self.value_mapping.mapping(item["value"])
        else:
            value = item["value"]

        return {
            "prompt": prompt,
            "response": item["response"],
            "input": _input,
            "value": value,
            "index": item["index"],
        }


class Attempt2ValueRewardModelingDatasetV2(ComposeDatasetMixin):
    # Update from `Attempt2ValueRewardModelingDataset`: @2024/01/08
    #   Remove duplicate responses, including intermediate states.
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str, reader: Callable, max_value: int,
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False,
                 value_mapping: Optional[Value2LabelMapping] = None,
                 remove_full_response: bool = False):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        inter_states = reader(file_path)
        self.id2inter_states = collections.defaultdict(list)
        for item in inter_states:
            self.id2inter_states[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        data = []
        abandoned = []
        logs = {}
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

                    response_set = set()
                    for state in inter_item["inter_states"]:
                        if "value" in state:
                            response = state["state"]
                            if response in response_set:
                                continue
                            response_set.add(response)
                            value = state["value"]
                            new_item = copy.deepcopy(item)
                            new_item["response"] = response
                            new_item["value"] = value
                            new_item["index"] = item_id if not re_index else len(data)
                            data.append(new_item)
                    if remove_full_response:
                        continue
                    for full_response in inter_item["response"]:
                        if full_response in response_set:
                            continue
                        response_set.add(full_response)
                        value = parse_leaf_node_value(full_response, item["label"], logs) * max_value
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
        self.value_mapping = value_mapping

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        prompt, _input = self.compose_input(item, item["response"])

        if self.value_mapping is not None:
            value = self.value_mapping.mapping(item["value"])
        else:
            value = item["value"]

        return {
            "prompt": prompt,
            "response": item["response"],
            "input": _input,
            "value": value,
            "index": item["index"],
        }


class Attempt2ValueRewardModelingDatasetV3(ComposeDatasetMixin):
    # Update from `Attempt2ValueRewardModelingDatasetV2`: @2024/02/01
    #   Add balance option to balance the samples from positive or negative responses.
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 original_data_file: str, original_reader: Callable, template: str, reader: Callable, max_value: int,
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 format_filter: Optional[Callable] = None,
                 re_index: bool = False,
                 value_mapping: Optional[Value2LabelMapping] = None,
                 remove_full_response: bool = False):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)

        self.tokenizer = tokenizer

        inter_states = reader(file_path)
        self.id2inter_states = collections.defaultdict(list)
        for item in inter_states:
            self.id2inter_states[item["id"]].append(item)

        original_data = original_reader(original_data_file)
        pos_data = []
        neg_data = []
        abandoned = []
        logs = {}
        cnt = collections.Counter()
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

                    response_set = set()
                    for state in inter_item["inter_states"]:
                        if "value" in state:
                            response = state["state"]
                            if response in response_set:
                                continue
                            response_set.add(response)
                            value = state["value"]
                            new_item = copy.deepcopy(item)
                            new_item["response"] = response
                            new_item["value"] = value
                            new_item["index"] = item_id if not re_index else (len(pos_data) + len(neg_data))

                            resp_id = state["resp_id"]
                            parent_resp = inter_item["response"][resp_id]
                            if parse_leaf_node_value(parent_resp, item["label"], logs) == 1:
                                pos_data.append(new_item)
                            else:
                                neg_data.append(new_item)

                            cnt[value] += 1
                    if remove_full_response:
                        continue
                    for full_response in inter_item["response"]:
                        if full_response in response_set:
                            continue
                        response_set.add(full_response)
                        value = parse_leaf_node_value(full_response, item["label"], logs) * max_value
                        new_item = copy.deepcopy(item)
                        new_item["response"] = full_response
                        new_item["value"] = value
                        new_item["index"] = item_id if not re_index else (len(pos_data) + len(neg_data))
                        # data.append(new_item)
                        if value == max_value:
                            pos_data.append(new_item)
                        else:
                            neg_data.append(new_item)

                        cnt[value] += 1

        logger.info(f"Value distribution: {cnt}")

        logger.info(f"Attempt2ValueRewardModelingDataset: Positive {len(pos_data)} / {len(original_data)}")
        logger.info(f"Attempt2ValueRewardModelingDataset: Negative {len(neg_data)} / {len(original_data)}")
        # Balance the positive and negative samples.
        if len(pos_data) > len(neg_data):
            pos_data = random.sample(pos_data, len(neg_data))
        else:
            neg_data = random.sample(neg_data, len(pos_data))
        data = pos_data + neg_data
        logger.info(f"Attempt2ValueRewardModelingDataset: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys
        self.value_mapping = value_mapping

        if format_filter:
            logger.info(f"Abandoned some of non-format examples:\n{len(abandoned)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        prompt, _input = self.compose_input(item, item["response"])

        if self.value_mapping is not None:
            value = self.value_mapping.mapping(item["value"])
        else:
            value = item["value"]

        return {
            "prompt": prompt,
            "response": item["response"],
            "input": _input,
            "value": value,
            "index": item["index"],
        }


class CompleteTrajRewardModelingDataset(ComposeDatasetMixin):
    """
    This dataset only loads the complete trajectory.

    This dataset can also be used for training.
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
        logs = {}
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
                        new_item["value"] = parse_leaf_node_value(full_response, item["label"], logs)
                        data.append(new_item)

        logger.info(f"Attempt2ValueRewardModelingDataset: {len(data)} / {len(original_data)}")
        logger.info(logs)
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
            "value": item["value"],
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
    raise NotImplementedError()  # FIXME: Maybe we need to re-preprocess the input files from scratch for ReClor dataset. Because there are some lines are empty steps, e.g., Action 1: \n.

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


def extract_react_ending_positions_v2(tokenizer: PreTrainedTokenizer, response: str, max_seq_length: int):
    steps, step_types = process_response_v2(response)
    raw_lines = response.split("\n")
    endings = []
    ending_types = []
    resp_start = False
    for i, (step_id, step) in enumerate(steps):
        if not resp_start:
            if step.startswith("Thought 1:"):
                resp_start = True
            else:
                continue
        partial_traj = "\n".join(raw_lines[:(step_id + 1)])
        input_ids = tokenizer(partial_traj, truncation=True, max_length=max_seq_length)["input_ids"]
        endings.append(len(input_ids) - 1)
        ending_types.append(step_types[i])

    assert resp_start, response
    assert len(endings) > 0, (response, steps)
    assert len(endings) == len(process_response_v2(response[response.find("Thought 1:"):])[0]), (response, steps)
    return endings, ending_types


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
        step_types = []
        padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
        for b, item in enumerate(batch):
            ending, ending_types = extract_react_ending_positions_v2(self.tokenizer, item["input"], self.max_seq_length)  # FIXED: @2024/01/06 for ReClor.
            if self.tokenizer.padding_side == "left":
                ending = [e + padding_len[b].item() for e in ending]
            endings.append(ending)
            step_types.append(ending_types)
        assert len(step_types) == len(endings), (step_types, endings)

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "input": inputs,
            "response": [item["response"] for item in batch],
            "ending": endings,
            "type": step_types,
        }
        return encoded_inputs


# ========================== Compare Response

class CompareResponseReader(ComposeDatasetMixin):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer,
                 response_file_b: str, correct_intersection: bool,
                 original_data_file: str, original_reader: Callable, template: str,
                 instruction: str = "", few_shot_prompts: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 api_based: bool = False, ):
        super().__init__(template, instruction, few_shot_prompts, compose_keys)
        self.tokenizer = tokenizer

        original_data = original_reader(original_data_file)

        responses_a = json.load(open(file_path))
        responses_b = json.load(open(response_file_b))

        if correct_intersection:
            responses_a = self.filter_correct_responses(responses_a)
            responses_b = self.filter_correct_responses(responses_b)
            id_set_a = set([item["id"] for item in responses_a])
            id_set_b = set([item["id"] for item in responses_b])
            id_set = id_set_a.intersection(id_set_b)
            responses_a = [item for item in responses_a if item["id"] in id_set]
            responses_b = [item for item in responses_b if item["id"] in id_set]

        self.id2response_a = {item["id"]: item for item in responses_a}
        self.id2response_b = {item["id"]: item for item in responses_b}

        data = []
        for i, item in enumerate(original_data):
            if "index" in item:
                item_id = item["index"]
            else:
                item_id = i
            if item_id in self.id2response_a and item_id in self.id2response_b:
                new_item = copy.deepcopy(item)
                new_item["response_a"] = self.id2response_a[item_id]["response"]
                new_item["response_b"] = self.id2response_b[item_id]["response"]
                new_item["index"] = item_id
                data.append(new_item)

        logger.info(f"CompareResponseReader: {len(data)} / {len(original_data)}")
        self.data: List[Dict[str, Any]] = data
        self.template = template
        self.instruction = instruction
        self.few_shot_prompts = few_shot_prompts
        self.compose_keys = compose_keys
        self.api_based = api_based

    def filter_correct_responses(self, responses):
        outputs = []
        logs = {}
        for item in responses:
            value = parse_leaf_node_value(item["response"], item["label"], logs)
            if value == 1:
                outputs.append(item)
        logger.info(f"Filtering correct responses: {len(responses)} -> {len(outputs)}")
        logger.info(f"Logs: {logs}")
        return outputs

    def __len__(self):
        return len(self.data)

    def api_getitem(self, index):
        item = self.data[index]
        prompt, _ = self.compose_input(item, "")

        return {
            "text": prompt,
            "meta_data": {
                "index": item["index"],
                "label": -1,
                "text": prompt,
            }
        }

    def __getitem__(self, index):
        if self.api_based:
            return self.api_getitem(index)
        raise NotImplementedError
