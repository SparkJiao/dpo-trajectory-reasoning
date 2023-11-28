import copy
import json
import os.path
from typing import List, Dict, Tuple, Union, Any, Callable

from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)

templates = [
    "Answer the following question:\n\nContext:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\nLet's reasoning step by step:",
    "[Context]\n{}\n\n[Question]\n{}\n\n[Options]\n{}\n\nHere are the transformed ones in logic form:\n\n",
    "{}\n\nBased on the original description of the problem above and the corresponding logic form. What's the correct answer?\n",
    "{}\n\nThe answer is ",
    "[Context]\n{}\n\n[Question]\n{}\n\n[Options]\n{}\n\nPlease decompose the problem above into smaller ones so that we can solve it separately and reach the final answer by consideing each subproblem and merge the sub-conclusions.\n\n",
    "[Response]\n{}\n\n[Json]\n",
    "Context:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\n",
    "Context:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\n<Reasoning Start>\n",
    "Context:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\nThought 1:",
    "Context:\n{}\n\nQuestion:\n{}\n\nOptions:\n{}\n\nThought 1: {}"  # Continue from the previous one for response composition.
]


def read_single_file(file_path: str, suffix: str = ""):
    return open(file_path, "r").read().strip() + suffix


def _format_option_list(option_list: List[str], _rank2option: List[str]) -> str:
    res = []
    for op_id, op in enumerate(option_list):
        res.append(f"{_rank2option[op_id]}. {op}")
    return "\n".join(res)


class LogicQAReader:
    rank2option = ['A', 'B', 'C', 'D']

    def __init__(self, flat_options: bool = False, option_order: str = "ABCD"):
        self.flat_options = flat_options
        self.option_order = option_order

    def __call__(self, file):
        all_context = []
        all_question = []
        all_option_list = []
        all_label = []

        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                all_context.append(item["text"])
                all_question.append(item["question"])

                options = []
                ordered_label = -1
                for i, x in enumerate(self.option_order):
                    idx = ord(x) - ord('A')
                    options.append(item["options"][idx])

                    if ord(x) - ord('A') == item["answer"]:
                        ordered_label = i
                all_option_list.append(options)
                all_label.append(ordered_label)

        return [
            {
                "context": context,
                "question": question,
                "option_list": _format_option_list(option_list, self.rank2option) if self.flat_options else option_list,
                "label": label,
            } for context, question, option_list, label in zip(all_context, all_question, all_option_list, all_label)
        ]


class SubResponseMergeReader(LogicQAReader):
    def __init__(self, inter_states_file: str, flat_options: bool = False):
        super().__init__(flat_options)
        self.inter_states = json.load(open(inter_states_file, "r"))

    def __call__(self, file):
        """
        :param file: The original data of LogiQA-v2.
        :return:
        """
        original_data = super().__call__(file)
        id2original_data = {idx: item for idx, item in enumerate(original_data)}

        outputs = []
        for item in self.inter_states:
            idx = item["id"]
            # if idx not in id2original_data:
            #     continue
            assert idx in id2original_data, idx
            original_item = id2original_data[idx]
            for state_id, state in enumerate(item["inter_states"]):
                new_item = copy.deepcopy(original_item)
                if isinstance(state, str):
                    new_item["response"] = state
                else:
                    new_item["response"] = state["state"]
                new_item["id"] = f"{idx}_{state_id}"
                outputs.append(new_item)
        return outputs


class ComposePromptGenerator(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, read_func, template_id: int = 0,
                 instruction: str = "", few_shot_prompt: str = "",
                 compose_keys: Union[List, Tuple, ListConfig] = ("context", "question", "options"),
                 max_data_num: int = -1,
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None, ):
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.compose_keys = compose_keys
        self.input_data: List[Dict[str, Any]] = read_func(file_path)

        flushed_data = {}
        if flush_file is not None and os.path.exists(flush_file):
            tmp = open(flush_file, "r").readlines()
            for line in tmp:
                item = json.loads(line)
                if "response" in item:
                    if isinstance(item["response"], str):
                        if item["response"].strip() == "":
                            continue
                    elif isinstance(item["response"], list):
                        if any([tmp.strip() == "" for tmp in item["response"]]):
                            continue
                flushed_data[item["id"]] = item
            logger.info(f"Loaded flushed data: {len(flushed_data)}")

        self.inputs = []
        self.indices = []
        self.labels = []
        for i in range(len(self.input_data)):
            idx = self.input_data[i]["id"] if "id" in self.input_data[i] else i
            if idx in flushed_data:
                continue

            _input = ""
            if self.instruction:
                _input += self.instruction + "\n\n"
            if self.few_shot_prompt:
                _input += self.few_shot_prompt + "\n\n"

            params = [self.input_data[i][key] for key in self.compose_keys]
            _input += templates[template_id].format(*params)

            self.inputs.append(_input)
            # if "id" in self.input_data[i]:
            #     self.indices.append(self.input_data[i]["id"])
            # else:
            #     self.indices.append(i)
            self.indices.append(idx)
            self.labels.append(self.input_data[i]["label"])

        self.tokenizer = tokenizer
        self.max_data_num = max_data_num
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.inputs))
        return len(self.inputs)

    def api_getitem(self, index):
        return {
            "text": self.inputs[index],
            "meta_data": {
                "index": self.indices[index],
                "label": self.labels[index],
                "text": self.inputs[index],
            }
        }

    def service_getitem(self, index):
        prompt = self.inputs[index]
        response = self.service_processor(prompt)
        return {
            "input": prompt,
            "response": response,
            "meta_data": {
                "index": self.indices[index],
                "label": self.labels[index],
                "text": self.inputs[index],
                "response": response,
            }
        }

    def __getitem__(self, index):
        if self.api_based:
            return self.api_getitem(index)
        if self.service_based:
            return self.service_getitem(index)
        return {
            "input": self.inputs[index],
            "index": self.indices[index],
            "label": self.labels[index],
        }


class TextInputCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, padding: str = "longest",
                 pp_inputs_processor: Callable = None, **kwargs):
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.max_seq_length = max_seq_length
        self.padding = padding
        self.pp_inputs_processor = pp_inputs_processor

    def __call__(self, batch):
        inputs = [b["input"] for b in batch]
        index = [b["index"] for b in batch]
        labels = [b["label"] for b in batch]

        model_inputs = self.tokenizer(inputs, padding=self.padding, truncation=True, max_length=self.max_seq_length,
                                      return_tensors="pt")

        if self.pp_inputs_processor is not None:
            return self.pp_inputs_processor(model_inputs, self.tokenizer)

        model_inputs["labels"] = labels
        model_inputs["meta_data"] = {
            "inputs": inputs,
            "labels": labels,
            "index": index,
        }
        return model_inputs
