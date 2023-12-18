import copy
import json
import os.path
from typing import List, Dict, Tuple, Union, Any, Callable

from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from general_util.logger import get_child_logger
from data.logiqav2 import _format_option_list

logger = get_child_logger(__name__)


class ReClorReader:
    rank2option = ['A', 'B', 'C', 'D']

    def __init__(self, flat_options: bool = False, option_order: str = "ABCD"):
        self.flat_options = flat_options
        self.option_order = option_order

    def __call__(self, file):
        data = json.load(open(file, 'r'))

        all_context = []
        all_question = []
        all_option_list = []
        all_label = []
        for sample in data:
            all_context.append(sample["context"])
            all_question.append(sample["question"])

            options = []
            ordered_label = -1
            for i, x in enumerate(self.option_order):
                idx = ord(x) - ord('A')
                options.append(sample["answers"][idx])

                if "label" in sample and ord(x) - ord('A') == sample["label"]:
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
