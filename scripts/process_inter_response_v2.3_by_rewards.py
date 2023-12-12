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
    parser.add_argument("--chosen_r", type=float, default=5.0)
    parser.add_argument("--reject_l", type=float, default=0.0)
    parser.add_argument("--inter_state_file", type=str)
    parser.add_argument("--step_ratio_diff", type=float, default=0.4)
    parser.add_argument("--exclude_full", action="store_true", default=False)
    args = parser.parse_args()

    response2reward = {}
    rewards = json.load(open(args.input_file))
    duplicates = set()
    cnt = 0
    for item in rewards:
        if item["response"] in response2reward:
            duplicates.add(item["response"])
            cnt += 1
        assert len(item["reward"]) == 1
        response2reward[item["response"]] = item["reward"][0]

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)

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
    num_partial_longer_win = 0
    num_partial_shorter_win = 0
    num_cross_full_win = 0
    num_cross_part_win = 0
    for item in tqdm(inter_states, total=len(inter_states)):
        idx = item["id"]
        # if idx not in state_id2values:
        #     jumped += 1
        #     continue

        resp_id2states = collections.defaultdict(list)
        for s_id, s in enumerate(item["inter_states"]):
            # if s_id not in state_id2values[idx]:
            #     continue
            # s["value"] = state_id2values[idx][s_id][0]
            # assert state_id2values[idx][s_id][1] in s["state"], (state_id2values[idx][s_id][1], s["state"])
            if s["state"] not in response2reward:
                jumped += 1
                continue
            s["value"] = response2reward[s["state"]]
            resp_id2states[s["resp_id"]].append(s)

        for resp_id, states in resp_id2states.items():
            resp_id2states[resp_id] = sorted(states, key=lambda tmp: tmp["step_id"])

        all_rationales = []
        for resp_id, resp in enumerate(item["response"]):
            full_steps = len(process_response(resp))

            for state in resp_id2states[resp_id]:
                state["step_ratio"] = (state["step_id"] + 1) / full_steps
                state["is_full"] = False
                all_rationales.append(state)

            if resp not in response2reward:
                continue
            resp_full_state = {
                "state": resp,
                "value": parse_leaf_node_value(resp, item["label"]) * response2reward[resp],
                "step_id": full_steps - 1,
                "resp_id": resp_id,
                "is_full": True,
                "step_ratio": 1.0,
            }

            all_rationales.append(resp_full_state)

        pos = []
        neg = []
        for r in all_rationales:
            if r["value"] >= args.chosen_r:
                pos.append(r)
            elif r["value"] <= args.reject_l:
                neg.append(r)
        if len(pos) == 0 or len(neg) == 0:
            continue

        for r in pos:
            for r2 in neg:
                if args.exclude_full and r["is_full"] and r2["is_full"]:
                    continue
                # if r2["resp_id"] == r["resp_id"]:
                #     continue
                if abs(r["step_ratio"] - r2["step_ratio"]) > args.step_ratio_diff:
                    continue
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

    print(f"Jumped: {jumped}")
    print(len(outputs))
    print(f"Partial longer win: {num_partial_longer_win}")
    print(f"Partial shorter win: {num_partial_shorter_win}")
    print(f"Cross full win: {num_cross_full_win}")
    print(f"Cross partial win: {num_cross_part_win}")
    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
