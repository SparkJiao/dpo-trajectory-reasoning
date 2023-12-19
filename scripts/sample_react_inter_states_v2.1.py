import random
import re
import argparse
import json
from typing import List, Set

"""
Ver2.0 updates:
    1. In this version we want devise a way to sample more intermediate states.
    2. Remove the sampling of action states.
"""


def clean_react_response(response: str):
    if "Context:\n" in response:
        response = response.split("Context:\n")[0]

    lines = response.split("\n")
    finish_lines = [lines for line in lines if "Finish[The answer is" in line and line.startswith("Action ")]
    if len(finish_lines) != 1:
        return None

    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "Finish[The answer is" in line and line.startswith("Action "):
            break
    assert "Finish[The answer is" in new_lines[-1] and new_lines[-1].startswith("Action "), new_lines
    response = "\n".join(new_lines)
    response = response.strip()
    return response


def process_response(response: str):
    lines = response.split("\n")
    lines = list(filter(lambda x: x[1].startswith("Thought ") or x[1].startswith("Action ") or x[1].startswith("Observation "), enumerate(lines)))
    return lines


def get_type(line: str):
    if line.startswith("Thought"):
        return "Thought"
    elif line.startswith("Action"):
        return "Action"
    elif line.startswith("Observation"):
        return "Observation"
    else:
        raise ValueError(f"Unknown line: {line}")


def sample_intermediate_responses(responses: List[str], ratio_s: float, ratio: float, include_types: Set[str]):
    inter_states = []
    for resp_id, response in enumerate(responses):
        original_lines = response.split("\n")
        steps = process_response(response)
        steps = steps[:-1]  # Remove the `Final` action.

        start_step_id = int(len(steps) * ratio_s)
        for i in range(start_step_id, len(steps)):
            step_type = get_type(steps[i][1])
            if step_type not in include_types:
                continue
            if random.random() < ratio:
                inter_state = "\n".join(original_lines[:(steps[i][0] + 1)])
                inter_states.append({
                    "state": inter_state,
                    "step_id": i,
                    "resp_id": resp_id,
                    "type": step_type,
                })

    return inter_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--split_num", type=int, default=1)
    parser.add_argument("--ratio_s", type=float, default=0.2)
    parser.add_argument("--ratio", type=float, default=0.4)
    parser.add_argument("--include_types", default="{'Thought','Observation'}", type=str)
    args = parser.parse_args()

    data = json.load(open(args.input_file))
    args.include_types = eval(args.include_types)
    print(f"Include types: {args.include_types}")

    # Clean
    cleaned_data = []
    cnt_a = 0
    cnt_b = 0
    for item in data:
        response = item["response"]
        preds = item["pred"]
        if isinstance(response, str):
            response = [response]
            preds = [preds]
        new_responses = []
        new_preds = []
        for res, pred in zip(response, preds):
            cnt_a += 1
            res = clean_react_response(res)
            if res is None:
                continue
            new_responses.append(res)
            new_preds.append(pred)
        if len(new_responses) == 0:
            continue
        item["response"] = new_responses
        item["pred"] = new_preds
        cleaned_data.append(item)
        cnt_b += len(new_responses)

    print(f"Clean: {cnt_a} -> {cnt_b}")

    # Process
    t_cnt = 0
    a_cnt = 0
    o_cnt = 0
    total = 0
    for item in cleaned_data:
        item["inter_states"] = sample_intermediate_responses(item["response"], ratio_s=args.ratio_s, ratio=args.ratio, include_types=args.include_types)
        for state in item["inter_states"]:
            line_type = state["type"]
            if line_type == "Thought":
                t_cnt += 1
            elif line_type == "Action":
                a_cnt += 1
            elif line_type == "Observation":
                o_cnt += 1
            else:
                raise ValueError(f"Unknown line type: {line_type}")
        total += len(item["inter_states"])
    print(f"Thought: {t_cnt}, Action: {a_cnt}, Observation: {o_cnt}")
    print(f"Total: {total}")

    if args.split_num == 1:
        json.dump(cleaned_data, open(args.output_file, "w"), indent=2)
    else:
        split_size = len(cleaned_data) // args.split_num
        for i in range(args.split_num):
            json.dump(cleaned_data[i * split_size: (i + 1) * split_size], open(args.output_file.replace(".json", f".{i}-of-{args.split_num}.json"), "w"),
                      indent=2)


if __name__ == '__main__':
    main()
