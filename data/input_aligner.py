import copy
import json
import os.path
import random
from typing import Dict, List, Callable, Union
from glob import glob

from omegaconf.listconfig import ListConfig
from tqdm import tqdm

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def concat_aligner(aligners: List[Callable]):
    def func(data: List[Dict]):
        for aligner in aligners:
            data = aligner(data)
        return data

    return func


def accumulate_step_aligner(step_field: str = "accumulated_response", index_field: str = "id", correct_only: bool = False,
                            step_offset: int = 0):
    def func(data: List[Dict]):
        outputs = []
        s = set()
        for item in tqdm(data, desc="Accumulating steps", total=len(data)):
            steps = item[step_field]
            if step_offset > 0:
                steps = steps[:-step_offset]
            for i, step in enumerate(steps):
                item_copy = copy.deepcopy(item)
                if isinstance(step, dict):
                    g = step["id"].split("_")
                    resp_id = int(g[-2])
                    if correct_only:
                        if item["pred"][resp_id]:
                            if isinstance(item["label"], int):
                                if ord(item["pred"][resp_id]) - ord("A") == item["label"]:
                                    pass
                                else:
                                    continue
                            elif isinstance(item["label"], str):
                                if item["pred"][resp_id] == item["label"]:
                                    pass
                                else:
                                    continue
                            else:
                                raise ValueError(f"Unknown type of label: {type(item['label'])}")
                        else:
                            continue

                    if step["response"] in s:
                        continue
                    s.add(step["response"])
                    item_copy[step_field] = step["response"]
                    item_copy[index_field] = step["id"]
                elif isinstance(step, str):
                    if correct_only:
                        raise NotImplementedError("Not implemented yet")

                    if step in s:
                        continue
                    s.add(step)
                    item_copy[step_field] = step
                    item_copy[index_field] = f"{item[index_field]}_{i}"
                else:
                    raise ValueError(f"Unknown type of step: {type(step)}")

                outputs.append(item_copy)

        logger.info(f"Accumulated {len(outputs)} steps")

        return outputs

    return func


def meta_math_type_filter(dataset_type: Union[str, ListConfig]):
    def func(data: List[Dict]):
        types = list(dataset_type) if isinstance(dataset_type, ListConfig) else [dataset_type]

        outputs = []
        for item in data:
            if any(tmp in item["type"] for tmp in types):
                outputs.append(item)

        logger.info(f"Filtered {len(outputs)} items with types: {types}")

        return outputs

    return func


def completion_count_aligner(completion_file: str, index_field: str = "id", value_field: str = "value", reduction_ids: Union[List[int], ListConfig] = None):
    def func(data: List[Dict]):
        if os.path.exists(completion_file):
            completion_files = [completion_file]
        else:
            completion_files = glob(completion_file)
        completions = []
        for file in completion_files:
            completions += json.load(open(file))
        id2completion = {item["id"]: item for item in completions}
        if reduction_ids and isinstance(reduction_ids, ListConfig):
            _reduction_ids = list(reduction_ids)
        else:
            _reduction_ids = reduction_ids

        outputs = []
        for item in data:
            item_id = item[index_field]
            if item_id not in id2completion:
                continue
            completion_preds = id2completion[item_id]["pred"]
            label = id2completion[item_id]["label"]
            cnt = 0
            for pred in completion_preds:
                if isinstance(label, int):
                    if pred and ord(pred.strip()) - ord("A") == label:
                        cnt += 1
                elif isinstance(label, str):
                    if pred == label:
                        cnt += 1

            if _reduction_ids:
                if cnt in _reduction_ids:
                    item[value_field] = 1
                else:
                    item[value_field] = 0
            else:
                item[value_field] = cnt
            outputs.append(item)

        logger.info(f"Counted {len(outputs)} items")

        return outputs

    return func


def flatten_completion(remove_answer_clause: bool = True, ):
    def func(completions: List[str]):
        if remove_answer_clause:
            completions = [item.split("### The answer is")[0] for item in completions]

        text = "\n\n[POSSIBLE FOLLOWING SOLUTIONS]"
        for i, completion in enumerate(completions):
            text += f"\n({i}) {completion}"
        return text

    return func


def completion_flatten_aligner(completion_file: str,
                               index_field: str = "id",
                               value_field: str = "value",
                               reduction_ids: Union[List[int], ListConfig] = None,
                               flatten_fn: Callable = flatten_completion()):
    def func(data: List[Dict]):
        completions = json.load(open(completion_file))
        id2completion = {item["id"]: item for item in completions}
        if reduction_ids and isinstance(reduction_ids, ListConfig):
            _reduction_ids = list(reduction_ids)
        else:
            _reduction_ids = reduction_ids

        outputs = []
        for item in data:
            item_id = item[index_field]
            if item_id not in id2completion:
                continue
            completion_preds = id2completion[item_id]["pred"]
            label = id2completion[item_id]["label"]
            cnt = 0
            for pred in completion_preds:
                if pred and ord(pred.strip()) - ord("A") == label:
                    cnt += 1

            if _reduction_ids:
                if cnt in _reduction_ids:
                    item[value_field] = 1
                else:
                    item[value_field] = 0
            else:
                item[value_field] = cnt

            flatten_response = flatten_fn(id2completion[item_id]["response"])
            item["flatten_response"] = flatten_response

            outputs.append(item)

        logger.info(f"Counted {len(outputs)} items")

        return outputs

    return func


def dpo_pair_aligner_cleaned(response_field: str = "response",
                             id_field: str = "id",
                             do_sample: bool = False, ):
    """
    This aligner only accepts the cleaned file, which has removing all empty responses and combined with original data.
    :return: Callable
    """

    def func(data: List[Dict]):
        outputs = []
        for item in data:
            pos_resp = []
            neg_resp = []
            for i, (resp, pred) in enumerate(zip(item[response_field], item["pred"])):
                assert resp
                # assert pred
                if isinstance(resp, list):
                    assert isinstance(resp[0], str)
                    # assert "The answer is" in resp[-1], resp
                    resp = "".join(resp)

                if isinstance(item["label"], str):
                    if pred == item["label"]:
                        pos_resp.append((i, resp))
                    else:
                        neg_resp.append((i, resp))
                elif isinstance(item["label"], int):
                    if pred and ord(pred) - ord("A") == item["label"]:
                        pos_resp.append((i, resp))
                    else:
                        neg_resp.append((i, resp))
                else:
                    raise ValueError(f"Unknown type of label: {type(item['label'])}")

            if not (len(pos_resp) and len(neg_resp)):
                continue

            if do_sample:
                pos = random.choice(pos_resp)
                neg = random.choice(neg_resp)
                pos_resp = [pos]
                neg_resp = [neg]

            for pos in pos_resp:
                for neg in neg_resp:
                    new_item = copy.deepcopy(item)
                    new_item["pos"] = pos[1]
                    new_item["neg"] = neg[1]
                    new_item["pos_id"] = f"{item[id_field]}_{pos[0]}"
                    new_item["neg_id"] = f"{item[id_field]}_{neg[0]}"
                    outputs.append(new_item)

        logger.info(f"Counted {len(outputs)} DPO contrastive pairs.")
        return outputs

    return func


def deepseek_dpo_aligner(response_field: str = "response",
                         id_field: str = "id",
                         do_sample: bool = False,
                         eval_fn: str = "gsm8k"):
    from data.deepseek_math_utils.eval_script import eval_last_single_answer, eval_math

    eval_fns = {
        "gsm8k": eval_last_single_answer,
        "math": eval_math,
    }
    fn = eval_fns[eval_fn]

    def func(data: List[Dict]):
        outputs = []

        for item in data:
            r_set = set()
            pos_r = []
            neg_r = []
            for i, (resp, p) in enumerate(zip(item[response_field], item["pred"])):
                if resp in r_set:
                    continue

                r_set.add(resp)

                res = fn({"prediction": p, "answer": item["label"]})

                if res:
                    pos_r.append((i, resp))
                else:
                    neg_r.append((i, resp))

            if not (len(pos_r) and len(neg_r)):
                continue

            if do_sample:
                pos = random.choice(pos_r)
                neg = random.choice(neg_r)
                pos_r = [pos]
                neg_r = [neg]

            for pos in pos_r:
                for neg in neg_r:
                    new_item = copy.deepcopy(item)
                    new_item["pos"] = pos[1]
                    new_item["neg"] = neg[1]
                    new_item["pos_id"] = f"{item[id_field]}_{pos[0]}"
                    new_item["neg_id"] = f"{item[id_field]}_{neg[0]}"
                    outputs.append(new_item)

        logger.info(f"Counted {len(outputs)} DPO contrastive pairs.")
        return outputs

    return func


def deepseek_completion_count_aligner(completion_file: str,
                                      index_field: str = "id",
                                      value_field: str = "value",
                                      reduction_ids: Union[List[int], ListConfig] = None,
                                      eval_fn: str = "gsm8k"):
    from data.deepseek_math_utils.eval_script import eval_last_single_answer, eval_math

    eval_fns = {
        "gsm8k": eval_last_single_answer,
        "math": eval_math,
    }
    fn = eval_fns[eval_fn]

    def func(data: List[Dict]):
        if os.path.exists(completion_file):
            completion_files = [completion_file]
        else:
            completion_files = glob(completion_file)
        completions = []
        for file in completion_files:
            completions += json.load(open(file))
        logger.info(f"Loaded {len(completions)} completions")
        id2completion = {item["id"]: item for item in completions}
        if reduction_ids and isinstance(reduction_ids, ListConfig):
            _reduction_ids = list(reduction_ids)
        else:
            _reduction_ids = reduction_ids

        outputs = []
        missing = 0
        for item in tqdm(data, desc="Counting completions", total=len(data)):
            item_id = item[index_field]
            if item_id not in id2completion:
                missing += 1
                continue
            completion_preds = id2completion[item_id]["pred"]
            label = id2completion[item_id]["label"]
            cnt = 0
            for pred in completion_preds:
                res = fn({"prediction": pred, "answer": label})
                if res:
                    cnt += 1

            if _reduction_ids:
                if cnt in _reduction_ids:
                    item[value_field] = 1
                else:
                    item[value_field] = 0
            else:
                item[value_field] = cnt
            outputs.append(item)

        logger.info(f"Counted {len(outputs)} items")
        logger.info(f"Missing {missing} items")

        return outputs

    return func


def deepseek_orm_input_aligner(response_field: str = "response",
                               id_field: str = "id",
                               eval_fn: str = "gsm8k",
                               value_field: str = "value",
                               max_v: int = -1):
    assert max_v > 0, f"max_v should be greater than 0, but got {max_v}"
    from data.deepseek_math_utils.eval_script import eval_last_single_answer, eval_math

    eval_fns = {
        "gsm8k": eval_last_single_answer,
        "math": eval_math,
    }
    fn = eval_fns[eval_fn]

    def func(data: List[Dict]):
        outputs = []

        for item in data:
            r_set = set()
            pos_r = []
            neg_r = []
            for i, (resp, p) in enumerate(zip(item[response_field], item["pred"])):
                if resp in r_set:
                    continue

                r_set.add(resp)

                res = fn({"prediction": p, "answer": item["label"]})

                if res:
                    pos_r.append((i, resp))
                else:
                    neg_r.append((i, resp))

            for pos in pos_r:
                new_item = copy.deepcopy(item)
                new_item[response_field] = pos[1]
                new_item[id_field] = f"{item[id_field]}_{pos[0]}"
                new_item[value_field] = 1 * max_v
                outputs.append(new_item)

            for neg in neg_r:
                new_item = copy.deepcopy(item)
                new_item[response_field] = neg[1]
                new_item[id_field] = f"{item[id_field]}_{neg[0]}"
                new_item[value_field] = 0
                outputs.append(new_item)

        logger.info(f"Counted {len(outputs)} ORM inputs.")
        return outputs

    return func
