import argparse
import json
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm

"""
The output should come from `cot_clean.py`.
"""


def acc_func(item, response_field="response"):
    responses = item[response_field]
    preds = item["pred"]
    item_id = item["id"]

    acc_steps = []
    for resp_id, (resp, pred) in enumerate(zip(responses, preds)):
        acc = ""
        # for i, step in enumerate(resp):
        for i, step in enumerate(resp[:-2]):
            if "### The answer is" in step:
                break
            acc_resp = acc + step
            acc_id = f"{item_id}_{resp_id}_{i}"
            acc_steps.append({"id": acc_id, "response": acc_resp})
            acc += step

    item["accumulated_response"] = acc_steps
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--response_field", type=str, default="response")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    annotate = partial(acc_func, response_field=args.response_field)
    with Pool(args.num_workers) as p:
        data = list(tqdm(p.imap(annotate, data), total=len(data)))

    save_path = args.input_file.replace(".json", "_accumulated.json")
    json.dump(data, open(save_path, "w"))


if __name__ == '__main__':
    main()
