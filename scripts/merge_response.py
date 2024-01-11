import json
from glob import glob
import os
import argparse

"""
In this script, we simple merge the response, and use `construct_dpo_data_from_react_response_v1.1.py` to remove duplicate and calibrate the predictions.
"""


def merge_response(item_a, item_b):
    a_responses = item_a["response"]
    b_responses = item_b["response"]

    preds_a = item_a["pred"]
    preds_b = item_b["pred"]

    new_response = a_responses + b_responses
    new_pred = preds_a + preds_b

    assert item_a["id"] == item_b["id"]
    assert item_a["text"] == item_b["text"]
    assert item_a["label"] == item_b["label"]

    new_item = {
        "id": item_a["id"],
        "text": item_a["text"],
        "label": item_a["label"],
        "response": new_response,
        "pred": new_pred,
    }
    return new_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    print(files)

    data = []
    for file in files:
        data.extend(json.load(open(file)))
    print(f"Total number of data: ", len(data))

    id2data = {}
    for item in data:
        if item["id"] in id2data:
            id2data[item["id"]] = merge_response(id2data[item["id"]], item)
        else:
            id2data[item["id"]] = item
    print(f"Total number of data after merging: ", len(id2data))

    avg_resp_num = 0
    for item in id2data.values():
        avg_resp_num += len(item["response"])
    avg_resp_num /= len(id2data)
    print(f"Average number of responses: {avg_resp_num}")

    data = list(id2data.values())
    json.dump(data, open(args.output_file, "w"))


if __name__ == "__main__":
    main()
