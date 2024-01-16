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


class FOLIO2QAReader:
    rank2option = ['A', 'B']

    def __init__(self,):
        self.context = "There is one hypothesis and a group of premises:\n\nHypothesis:\n{}\n\nPremises:\n{}"
        self.question = "Verify the hypothesis is true or false based on the premises."
        self.option = "A. True\nB. False"

    def __call__(self, file):
        all_context = []
        all_option_list = []
        all_label = []
        with open(file) as f:
            for line in f.readlines():
                item = json.loads(line)

                conclusion = item["conclusion"]
                premises = item["premises"]
                premises_str = []
                for i, premise in enumerate(premises):
                    premises_str.append("{}. {}".format(i + 1, premise))
                premises_str = "\n".join(premises_str)
                label = 0 if item["label"] == "True" else 1

                all_context.append(self.context.format(conclusion, premises_str))
                all_option_list.append(self.option)
                all_label.append(label)

        return [
            {
                "context": context,
                "question": self.question,
                "option_list": option_list,
                "label": label,
            } for context, option_list, label in zip(all_context, all_option_list, all_label)
        ]


