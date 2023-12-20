import json
import argparse
from glob import glob
import os


def merge_inter_states_responses(responses, original_data):
    id2response = {}
    for item in responses:
        _id = item["id"]
        item_id, state_id = _id.split("_")
        id2response[f"{item_id}-{state_id}"] = item["response"].strip()

    for item in original_data:
        item_id = item["id"]
        for state_id, state in enumerate(item["inter_states"]):
            state["state_worsen"] = id2response[f"{item_id}-{state_id}"]

    return original_data


def merge_responses(responses, original_data):
    id2response = {}
    for item in responses:
        _id = item["id"]
        item_id, resp_id = _id.split("-")
        id2response[f"{item_id}-{resp_id}"] = item["response"].strip()

    for item in original_data:
        item_id = item["id"]
        worsen_responses = []
        for resp_id, _ in enumerate(item["response"]):
            if f"{item_id}-{resp_id}" in id2response:
                worsen_responses.append(id2response[f"{item_id}-{resp_id}"])
            else:
                worsen_responses.append("")
                print(f"Warning: {item_id}-{resp_id} not found")
        item["worsen_responses"] = worsen_responses

    return original_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--original_file", type=str, required=True)
    parser.add_argument("--is_inter_states", action="store_true", default=False)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        input_files = [args.input_file]
    else:
        input_files = glob(args.input_file)
    print(f"Input files:\n{input_files}")
    responses = []
    for files in input_files:
        responses.extend(json.load(open(files, "r")))

    if os.path.join(args.original_file, "data.jsonl"):
        original_files = [args.original_file]
    else:
        original_files = glob(os.path.join(args.original_file, "data.jsonl"))
    print(f"Original files:\n{original_files}")
    original_data = []
    for files in original_files:
        original_data.extend(json.load(open(files, "r")))

    if args.is_inter_states:
        output_data = merge_inter_states_responses(responses, original_data)
    else:
        output_data = merge_responses(responses, original_data)

    json.dump(output_data, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
