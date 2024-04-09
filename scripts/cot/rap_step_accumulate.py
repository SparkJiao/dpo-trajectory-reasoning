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

    s = set()
    acc_steps = []
    for resp_id, (resp, pred) in enumerate(zip(responses, preds)):
        raw_steps = resp.strip().split("\n")
        steps = [raw_steps[0]]
        for line in raw_steps[1:]:
            if line.replace("#", "").strip() == "":
                continue
            if not (line.startswith("SubQuestion ") or line.startswith("Answer ")):
                steps[-1] += "\n" + line
            else:
                steps.append(line)

        acc = ""
        for i, step in enumerate(steps[:-2]):
            if i == 0:
                acc_step = acc + step
            else:
                acc_step = acc + "\n" + step

            acc = acc_step

            if acc_step in s:
                continue

            s.add(acc_step)
            acc_id = f"{item_id}_{resp_id}_{i}"
            acc_steps.append({"id": acc_id, "response": acc_step})

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

    outputs = [item for item in data if item["accumulated_response"]]
    print(f"Filtered {len(outputs)} items")
    save_path = args.input_file.replace(".json", "_accumulated.json")
    json.dump(outputs, open(save_path, "w"))


if __name__ == '__main__':
    main()
