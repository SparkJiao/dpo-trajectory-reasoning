import argparse
import collections
import json
import os
from glob import glob
from tqdm import tqdm
import re

"""
Version 2.1

In this version, we will also include the original full-length response to enable full-partial comparison.

Version 2.3

In this version, we will control the step ratio difference between two states to be compared.

Version 2.4

Use negative threshold to filter out the longer sequences where the former steps have already possibly been wrong.
"""


def parse_leaf_node_value(response: str, label: int):
    groups = response.split("Finish")
    if len(groups) < 2:
        # print(f"Warning: Not a valid response: {response}")
        return 0
    response = groups[1]
    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        return 0
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            return 0


def process_response(response: str):
    lines = response.split("\n")
    lines = list(filter(lambda x: x[1].startswith("Thought ") or x[1].startswith("Action ") or x[1].startswith("Observation "), enumerate(lines)))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--diff", type=float, default=2.4)
    parser.add_argument("--inter_state_file", type=str)
    parser.add_argument("--step_ratio_diff", type=float, default=0.4)
    parser.add_argument("--negative_threshold", type=float, default=1.0)
    parser.add_argument("--early_stop", action="store_true", default=False)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    print(files)

    state_id2values = collections.defaultdict(dict)
    for file in files:
        data = json.load(open(file, "r"))

        for item in data:
            idx = item["id"]
            idx, state_id = idx.split("_")
            v = 0
            flag = True
            for res, p in zip(item["response"], item["pred"]):
                if res == "":
                    flag = False
                    break
                if p == "":
                    continue
                if ord(p.strip()) - ord("A") == item["label"]:
                    v += 1
            if not flag:
                continue
            state_id2values[int(idx)][int(state_id)] = (v, item["text"].split("Thought 1:")[1].strip())

    print(len(state_id2values))

    if os.path.exists(args.inter_state_file):
        inter_state_files = [args.inter_state_file]
    else:
        inter_state_files = glob(args.inter_state_file)
    print(inter_state_files)
    inter_states = []
    for state_file in inter_state_files:
        inter_states.extend(json.load(open(state_file, "r")))

    outputs = []
    jumped = 0
    masked = 0
    num_partial_longer_win = 0
    num_partial_shorter_win = 0
    num_cross_full_win = 0
    num_cross_part_win = 0
    for item in tqdm(inter_states, total=len(inter_states)):
        idx = item["id"]
        if idx not in state_id2values:
            jumped += 1
            continue

        for s_id, s in enumerate(item["inter_states"]):
            if s_id not in state_id2values[idx]:
                continue
            s["value"] = state_id2values[idx][s_id][0]
            assert state_id2values[idx][s_id][1] in s["state"], (state_id2values[idx][s_id][1], s["state"])

        resp_id2states = collections.defaultdict(list)
        for x_id, x in enumerate(item["inter_states"]):
            if x_id not in state_id2values[idx]:
                continue
            resp_id2states[x["resp_id"]].append(x)

        for resp_id, states in resp_id2states.items():
            resp_id2states[resp_id] = sorted(states, key=lambda tmp: tmp["step_id"])

        all_rationales = []
        for resp_id, resp in enumerate(item["response"]):
            error = False
            full_steps = len(process_response(resp))

            for state_id, state in enumerate(resp_id2states[resp_id]):
                state["step_ratio"] = (state["step_id"] + 1) / full_steps
                state["is_full"] = False

                if state_id > 0 and resp_id2states[resp_id][state_id - 1]["value"] <= args.negative_threshold:
                    if state["value"] > resp_id2states[resp_id][state_id - 1]["value"]:
                        masked += 1
                    state["value"] = min(state["value"], resp_id2states[resp_id][state_id - 1]["value"])
                    error = True

                all_rationales.append(state)

                if error and args.early_stop:
                    break

            if error and args.early_stop:
                continue

            resp_full_state = {
                "state": resp,
                "value": 3 * parse_leaf_node_value(resp, item["label"]),
                "step_id": full_steps - 1,
                "resp_id": resp_id,
                "is_full": True,
                "step_ratio": 1.0,
            }

            if len(resp_id2states[resp_id]) > 0 and resp_id2states[resp_id][-1]["value"] <= args.negative_threshold:
                if resp_full_state["value"] > resp_id2states[resp_id][-1]["value"]:
                    masked += 1
                resp_full_state["value"] = min(resp_full_state["value"], resp_id2states[resp_id][-1]["value"])

            all_rationales.append(resp_full_state)

        for r_i, r in enumerate(all_rationales):
            for r_j, r2 in enumerate(all_rationales[r_i + 1:]):
                if abs(r["step_ratio"] - r2["step_ratio"]) > args.step_ratio_diff:
                    continue
                if r["value"] - r2["value"] >= args.diff:
                    if r["step_ratio"] < r2["step_ratio"]:
                        num_partial_shorter_win += 1
                    elif r["step_ratio"] > r2["step_ratio"]:
                        num_partial_longer_win += 1
                    if r["is_full"] and not r2["is_full"]:
                        num_cross_full_win += 1
                    elif not r["is_full"] and r2["is_full"]:
                        num_cross_part_win += 1
                    outputs.append({
                        "id": idx,
                        "chosen": r["state"],
                        "reject": r2["state"],
                        "chosen_full": r["is_full"],
                        "reject_full": r2["is_full"],
                        "val_diff": r["value"] - r2["value"],
                    })
                elif r["value"] - r2["value"] <= -args.diff:
                    if r["step_ratio"] < r2["step_ratio"]:
                        num_partial_longer_win += 1
                    elif r["step_ratio"] > r2["step_ratio"]:
                        num_partial_shorter_win += 1
                    if not r["is_full"] and r2["is_full"]:
                        num_cross_full_win += 1
                    elif r["is_full"] and not r2["is_full"]:
                        num_cross_part_win += 1
                    outputs.append({
                        "id": idx,
                        "chosen": r2["state"],
                        "reject": r["state"],
                        "chosen_full": r2["is_full"],
                        "reject_full": r["is_full"],
                        "val_diff": r2["value"] - r["value"],
                    })

    print(f"Jumped: {jumped}")
    print(len(outputs))
    print(f"Masked: {masked}")
    print(f"Partial longer win: {num_partial_longer_win}")
    print(f"Partial shorter win: {num_partial_shorter_win}")
    print(f"Cross full win: {num_cross_full_win}")
    print(f"Cross partial win: {num_cross_part_win}")

    full_pairs = 0
    for x in outputs:
        if x["chosen_full"] and x["reject_full"]:
            full_pairs += 1
    print(f"Full pairs: {full_pairs}")

    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()