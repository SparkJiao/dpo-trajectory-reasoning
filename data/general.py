import copy
import json
import os.path
from typing import List, Dict, Tuple, Union, Any, Callable

from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger
from data.logiqav2 import LogicQAReader
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
