import json
import argparse
from glob import glob
import os
import re


def parse_leaf_node_value(response: str, label: int):
    groups = response.split("Finish")
    if len(groups) < 2:
        # print(f"Warning: Not a valid response: {response}")
        return 0
    response = groups[1]
    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        return 0
    elif len(preds) > 1:
        return 0
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            return 0


def merge_inter_states_responses(responses, original_data, filter_wrong=True):
    id2response = {}
    for item in responses:
        _id = item["id"]
        item_id, state_id = _id.split("_")
        id2response[f"{item_id}-{state_id}"] = item["response"].strip()

    outputs = []
    for item in original_data:
        item_id = item["id"]
        for state_id, state in enumerate(item["inter_states"]):
            resp_id = state["resp_id"]
            resp = item["response"][resp_id]
            v = parse_leaf_node_value(resp, item["label"])

            if filter_wrong and v == 0:
                continue

            if f"{item_id}-{state_id}" in id2response and id2response[f"{item_id}-{state_id}"].strip() != "":
                outputs.append({
                    "chosen": state["state"],
                    "reject": id2response[f"{item_id}-{state_id}"],
                    "id": item_id,
                    "is_full": False,
                })

    return outputs


def merge_responses(responses, original_data, filter_wrong=True):
    id2response = {}
    for item in responses:
        _id = item["id"]
        item_id, resp_id = _id.split("-")
        id2response[f"{item_id}-{resp_id}"] = item["response"].strip()

    outputs = []
    for item in original_data:
        item_id = item["id"]
        for resp_id, resp in enumerate(item["response"]):
            if f"{item_id}-{resp_id}" in id2response:
                worsen_resp = id2response[f"{item_id}-{resp_id}"]
            else:
                print(f"Warning: {item_id}-{resp_id} not found")
                continue

            v = parse_leaf_node_value(resp, item["label"])
            if filter_wrong and v == 0:
                continue

            outputs.append({
                "chosen": resp,
                "reject": worsen_resp,
                "id": item_id,
                "is_full": True,
            })

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--original_file", type=str, required=True)
    parser.add_argument("--is_inter_states", action="store_true", default=False)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--keep_wrong", action="store_true", default=False)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = glob(args.input_file)
    print(f"Input files:\n{input_files}")
    responses = []
    for files in input_files:
        responses.extend(json.load(open(files, "r")))

    if os.path.exists(args.original_file):
        original_files = [args.original_file]
    else:
        original_files = glob(args.original_file)
    print(f"Original files:\n{original_files}")
    original_data = []
    for files in original_files:
        original_data.extend(json.load(open(files, "r")))
    print(f"Original data: {len(original_data)}")

    if args.is_inter_states:
        output_data = merge_inter_states_responses(responses, original_data, filter_wrong=not args.keep_wrong)
    else:
        output_data = merge_responses(responses, original_data, filter_wrong=not args.keep_wrong)

    print(f"Output data: {len(output_data)}")
    json.dump(output_data, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
