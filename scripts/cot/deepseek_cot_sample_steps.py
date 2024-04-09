import json
import argparse
import os
from glob import glob
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


def acc_func(item, response_field: str = "response", offset: int = 0):
    s = set()
    acc_steps = []
    for i, (resp, p) in enumerate(zip(item[response_field], item["pred"])):
        steps = resp.split("\n")
        acc = ""
        if offset > 0:
            steps = steps[:-offset]
        for j, step in enumerate(steps):
            if j == 0:
                acc = step
            else:
                acc += "\n" + step

            if acc in s:
                continue

            s.add(acc)
            acc_steps.append({"id": f"{item['id']}_{i}_{j}", "response": acc})

    item["accumulated_response"] = acc_steps
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    data = []
    for file in files:
        data += json.load(open(file, "r"))

    annotate = partial(acc_func, response_field="response", offset=args.offset)
    with Pool(args.num_workers) as p:
        data = list(tqdm(p.imap(annotate, data), total=len(data)))

    outputs = [item for item in data if "accumulated_response" in item and item["accumulated_response"]]
    print(f"Number of items with accumulated responses: {len(outputs)}")
    json.dump(outputs, open(args.input_file.replace(".json", f"_accumulated_off{args.offset}.json"), "w"))


if __name__ == '__main__':
    main()
