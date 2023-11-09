import collections
import json
import os
import re
from typing import Dict, Any, List, Tuple
import numpy as np


class MCQAAnswerClean:
    def __init__(self, prompt: str = "zero-shot"):
        self.prompt = prompt

    def __call__(self, pred: str):
        # print("pred_before: ", pred)
        preds = re.findall(r"A|B|C|D|E", pred)
        if len(preds) == 0:
            return ""

        if self.prompt == "zero-shot":
            return preds[0]
        if self.prompt == "few-shot":
            return preds[-1]
        return preds[0]


class SeparatorClean:
    def __init__(self, separator: str = "Finish", separate_idx: int = 1, regrex: str = "A|B|C|D"):
        self.separator = separator
        self.separate_idx = separate_idx
        self.regrex = re.compile(regrex)

    def __call__(self, pred: str):
        preds = pred.split(self.separator)
        if len(preds) == 0:
            return ""

        pred = preds[self.separate_idx]
        preds = re.findall(self.regrex, pred)
        if len(preds) == 0:
            return ""
        return preds[0]


class ReActSeparatorClean:
    def __init__(self, separator: str = "Context:", separate_idx: int = 0, regrex: str = "A|B|C|D"):
        self.separator = separator  # Use for remove generated dummy examples
        self.separate_idx = separate_idx
        self.regrex = re.compile(regrex)

    def __call__(self, pred: str):
        if self.separator in pred:
            groups = pred.split(self.separator)
            pred = groups[self.separate_idx]

        if "Finish" in pred:
            pred = pred.split("Finish")[1]
            preds = re.findall(self.regrex, pred)
            if len(preds) == 0:
                return ""
            return preds[0]

        preds = re.findall(self.regrex, pred)
        if len(preds) == 0:
            return ""
        return preds[-1]


class BinaryAnswerClean:
    def __init__(self, prompt: str = "zero-shot"):
        self.prompt = prompt

    def __call__(self, pred: str):
        preds = re.findall(r"Yes|No", pred)
        if len(preds) == 0:
            return ""

        if self.prompt == "zero-shot":
            return preds[0]
        if self.prompt == "few-shot":
            return preds[-1]
        return preds[0]


class OpenAICallBack:
    def __init__(self, output_file: str, answer_clean: MCQAAnswerClean):
        self.predictions = []
        self.output_file = output_file
        self.answer_clean = answer_clean

        logging_file = output_file.replace(".json", ".jsonl")
        if os.path.exists(logging_file):
            with open(logging_file, "r") as f:
                for line in f.readlines():
                    self.predictions.append(json.loads(line))
            self.fw = open(logging_file, "a")
        else:
            self.fw = open(logging_file, "w")

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], **kwargs):
        text = meta_data["text"]
        if "label" in meta_data:
            label = meta_data["label"]
        else:
            label = -1
        index = meta_data["index"]
        # assert isinstance(index, str), type(index)
        # assert isinstance(text, str), type(text)
        # assert isinstance(label, int), type(label)

        response = batch_model_outputs["response"]
        if isinstance(response, str):
            pred_clean = self.answer_clean(response)
        elif isinstance(response, list):
            pred_clean = [self.answer_clean(item) for item in response]
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")
        # print("pred_after: ", pred_clean)
        self.predictions.append({
            "text": text,
            "label": label,
            "response": response,
            "pred": pred_clean,
            "id": index,
        })
        self.fw.write(json.dumps(self.predictions[-1]) + "\n")
        self.fw.flush()

    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        json.dump(self.predictions, open(self.output_file, "w"))
        self.fw.close()

        cnt = 0
        outputs = []
        for item in self.predictions:
            if isinstance(item["pred"], list):
                preds = item["pred"]
            else:
                preds = [item["pred"]]

            pred = collections.Counter(preds).most_common(1)[0][0]

            if not pred.strip():
                outputs.append(0)
                continue
            if len(pred.strip()) > 1:
                outputs.append(0)
                continue
            if isinstance(item["label"], str):
                if item["label"].strip() == pred.strip():
                    cnt += 1
            elif isinstance(item["label"], list) and isinstance(item["label"][0], str):
                if item["label"][0].strip() == pred.strip():
                    cnt += 1
            else:
                if item["label"] == ord(pred.strip()) - ord("A"):
                    cnt += 1
            outputs.append(ord(pred.strip()) - ord("A"))
        assert len(outputs) == len(self.predictions)

        np_output_file = self.output_file.replace(".json", ".npy")
        np.save(np_output_file, np.array(outputs))
        return {"acc": cnt / len(self.predictions)}, []
