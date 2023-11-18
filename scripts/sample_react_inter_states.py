import random
import re
import argparse
import json
from typing import List


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


def sample_intermediate_responses(responses: List[str]):
    inter_states = []
    for response in responses:
        lines = process_response(response)

        step = len(lines) // 2
        step = lines[step][0]
        inter_state = "\n".join(response.split("\n")[:(step + 1)])
        inter_states.append(inter_state)

    return inter_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--split_num", type=int, default=1)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

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
        item["inter_states"] = sample_intermediate_responses(item["response"])
        for state in item["inter_states"]:
            lines = state.split("\n")
            line = lines[-1]
            if line.startswith("Thought"):
                t_cnt += 1
            elif line.startswith("Action"):
                a_cnt += 1
            elif line.startswith("Observation"):
                o_cnt += 1
            else:
                print(f"Unknown line: {line}")
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
