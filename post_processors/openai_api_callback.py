import collections
import json
import os
import re
from typing import Dict, Any, List, Tuple, Union
import numpy as np
import vllm
from data import math_util
from data.deepseek_math_utils import eval_script, answer_extraction

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class PlaceholderClean:
    def __call__(self, pred: str):
        return "A"


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

        if len(preds) <= self.separate_idx:
            return ""

        pred = preds[self.separate_idx]
        preds = re.findall(self.regrex, pred)
        if len(preds) == 0 or len(preds) > 1:
            return ""
        return preds[0]


class ReActSeparatorClean:  # FIXED@2024-01-03: Add hard constraint.
    def __init__(self, separator: str = "Context:", separate_idx: int = 0, regrex: str = "A|B|C|D"):
        self.separator = separator  # Use for remove generated dummy examples
        self.separate_idx = separate_idx
        self.regrex = re.compile(regrex)

    def __call__(self, pred: str):
        if self.separator in pred:
            groups = pred.split(self.separator)
            pred = groups[self.separate_idx]

        if "Finish[" in pred:
            pred = pred.split("Finish[")[1]
            pred = pred.split("]")[0]
            preds = re.findall(self.regrex, pred)
            if len(preds) == 0:
                return ""
            elif len(preds) == 1:
                return preds[0]
            else:
                return ""  # FIXED@2023-12-27: To avoid the case where the large language models tends to generate multiple predictions to hack the answer.

        # preds = re.findall(self.regrex, pred)
        # if len(preds) == 0:
        #     return ""
        # return preds[-1]
        return ""


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
    def __init__(self, output_file: str, answer_clean: Union[MCQAAnswerClean, str], resume: bool = False, index_field: str = "index",
                 label_field: str = "label"):
        self.predictions = []
        self.output_file = output_file
        self.answer_clean = answer_clean
        self.index_field = index_field
        self.label_field = label_field

        logging_file = output_file.replace(".json", ".jsonl")
        if os.path.exists(logging_file):
            if resume:
                with open(logging_file, "r") as f:
                    for line in f.readlines():
                        # self.predictions.append(json.loads(line))
                        item = json.loads(line)
                        if isinstance(item["response"], str):
                            if item["response"].strip() == "":
                                continue
                        elif isinstance(item["response"], list):
                            if any([tmp.strip() == "" for tmp in item["response"]]):
                                continue
                        self.predictions.append(item)
                logger.info(f"Load {len(self.predictions)} from {logging_file}")
            self.fw = open(logging_file, "a")
        else:
            self.fw = open(logging_file, "w")

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], **kwargs):
        text = meta_data["text"]
        if self.label_field in meta_data:
            label = meta_data[self.label_field]
        else:
            label = -1
        index = meta_data[self.index_field]

        response = batch_model_outputs["response"]
        if isinstance(response, vllm.RequestOutput):
            if response.finished:
                response = [o.text for o in response.outputs]
                if len(response) == 1:
                    response = response[0]
            else:
                response = ""
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
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)
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
                outputs.append((item["id"], 0))
                continue
            if len(pred.strip()) > 1:
                outputs.append((item["id"], 0))
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
            outputs.append((item["id"], ord(pred.strip()) - ord("A")))
        assert len(outputs) == len(self.predictions)

        # Remove duplicated ids to satisfy the submission requirements of ReClor.
        outputs = sorted(outputs, key=lambda x: x[0])
        id_set = set()
        new_outputs = []
        for item in outputs:
            if item[0] not in id_set:
                new_outputs.append(item[1])
                id_set.add(item[0])
        outputs = new_outputs

        np_output_file = self.output_file.replace(".json", ".npy")
        np.save(np_output_file, np.array(outputs))

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "correct": cnt, "total": len(self.predictions)}
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        # return {"acc": cnt / len(self.predictions)}, []
        return metrics, []


class OpenAIMATHCallBack(OpenAICallBack):
    def get_results(self):
        save_dir = os.path.dirname(self.output_file)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)
        self.fw.close()

        cnt = 0
        outputs = []
        for item in self.predictions:
            if isinstance(item["pred"], list):
                preds = item["pred"]
            else:
                preds = [item["pred"]]

            pred = collections.Counter(preds).most_common(1)[0][0]

            res = math_util.is_equiv(pred, item["label"])
            if res:
                cnt += 1
            outputs.append((item["id"], res))

        assert len(outputs) == len(self.predictions)

        # Remove duplicated ids to satisfy the submission requirements of ReClor.
        outputs = sorted(outputs, key=lambda x: x[0])
        id_set = set()
        new_outputs = []
        for item in outputs:
            if item[0] not in id_set:
                new_outputs.append(item[1])
                id_set.add(item[0])
        outputs = new_outputs

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "correct": cnt, "total": len(self.predictions)}
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []


class DeepSeekMathCallBack(OpenAICallBack):
    eval_fns = {
        "gsm8k": eval_script.eval_last_single_answer,
        "math": eval_script.eval_math,
    }

    extract_fns = {
        "gsm8k": answer_extraction.extract_last_single_answer,
        "math": answer_extraction.extract_math_answer,
    }

    def __init__(self, *args, eval_fn: str = "gsm8k", **kwargs):
        super().__init__(*args, **kwargs, )
        self.eval_fn = self.eval_fns[eval_fn]
        if self.answer_clean in self.extract_fns:
            self.extract_fn = self.extract_fns[self.answer_clean]
        else:
            self.extract_fn = self.answer_clean

    def __call__(self, meta_data: Dict[str, Any], batch_model_outputs: Dict[str, Any], **kwargs):
        text = meta_data["text"]
        if self.label_field in meta_data:
            label = meta_data[self.label_field]
        else:
            label = -1
        index = meta_data[self.index_field]

        response = batch_model_outputs["response"]
        if isinstance(response, vllm.RequestOutput):
            if response.finished:
                response = [o.text for o in response.outputs]
                if len(response) == 1:
                    response = response[0]
            else:
                response = ""
        if isinstance(response, str):
            # pred_clean = self.answer_clean(response)
            pred_clean = self.extract_fn(text, response, "cot")
        elif isinstance(response, list):
            # pred_clean = [self.answer_clean(item) for item in response]
            pred_clean = [self.extract_fn(text, item, "cot") for item in response]
        else:
            raise ValueError(f"Unknown type of response: {type(response)}")

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

        self.fw.close()

        cnt = 0
        outputs = []
        for item in self.predictions:
            # if isinstance(item["pred"], list):
            #     preds = item["pred"]
            # else:
            #     preds = [item["pred"]]
            if isinstance(item["response"], list):
                preds = item["pred"]
            else:
                preds = [item["pred"]]

            if len(preds) > 0:
                if isinstance(preds[0], list):
                    pred = preds[0]  # TODO: How to add self-consistency for MATH dataset given the answer could be a list?
                else:
                    pred = collections.Counter(preds).most_common(1)[0][0]
                if pred is None:
                    pred = ""

                # res = math_util.is_equiv(pred, item["label"])
                # https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/infer/run_pal_eval.py#L168
                # https://github.com/deepseek-ai/DeepSeek-Math/blob/main/evaluation/infer/run_cot_eval.py#L120
                res = self.eval_fn({"prediction": pred, "answer": item["label"]})  # For CoT eval, use `prediction`, for Pal eval, use `program_output`.
            else:
                res = 0

            item["res"] = res

            if res:
                cnt += 1
            outputs.append((item["id"], res))

        assert len(outputs) == len(self.predictions)
        json.dump(self.predictions, open(self.output_file, "w"), indent=2)

        # Remove duplicated ids to satisfy the submission requirements of ReClor.
        outputs = sorted(outputs, key=lambda x: x[0])
        id_set = set()
        new_outputs = []
        for item in outputs:
            if item[0] not in id_set:
                new_outputs.append(item[1])
                id_set.add(item[0])
        outputs = new_outputs

        if len(self.predictions) == 0:
            metrics = {"acc": 0, "correct": 0, "total": 0}
        else:
            metrics = {"acc": cnt / len(self.predictions), "correct": cnt, "total": len(self.predictions)}
        json.dump(metrics, open(self.output_file.replace(".json", ".metrics.json"), "w"), indent=2)
        return metrics, []
