import argparse
import collections
import json
import os
from glob import glob
from tqdm import tqdm
import re
import random

"""
Version 2.1

In this version, we will also include the original full-length response to enable full-partial comparison.

Version 2.3

In this version, we will control the step ratio difference between two states to be compared.


Version 2.4

Use negative threshold to filter out the longer sequences where the former steps have already possibly been wrong.

Version 2.5

Add step_id clipping to filter out too short sequences.

Version 2.6

Discriminate the full examples.
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


def iterate_comparison_pair(idx, chosen_list, reject_list, step_ratio_diff, negative_sample_num: int):
    outputs = []
    statis = {
        "num_partial_longer_win": 0,
        "num_partial_shorter_win": 0,
        "num_cross_full_win": 0,
        "num_cross_part_win": 0,
    }
    for chosen in chosen_list:
        tmp = []
        for reject in reject_list:
            if abs(chosen["step_ratio"] - reject["step_ratio"]) > step_ratio_diff:
                continue
            if chosen["step_ratio"] < reject["step_ratio"]:
                # num_partial_shorter_win += 1
                statis["num_partial_shorter_win"] += 1
            elif chosen["step_ratio"] > reject["step_ratio"]:
                # num_partial_longer_win += 1
                statis["num_partial_longer_win"] += 1
            if chosen["is_full"] and not reject["is_full"]:
                # num_cross_full_win += 1
                statis["num_cross_full_win"] += 1
            elif not chosen["is_full"] and reject["is_full"]:
                # num_cross_part_win += 1
                statis["num_cross_part_win"] += 1
            tmp.append({
                "id": idx,
                "chosen": chosen["state"],
                "reject": reject["state"],
                "chosen_full": chosen["is_full"],
                "reject_full": reject["is_full"],
                "val_diff": chosen["value"] - reject["value"],
            })
        if len(tmp) > 0:
            outputs.extend(random.sample(tmp, min(len(tmp), negative_sample_num)))

    return outputs, statis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--diff", type=float, default=2.4)
    parser.add_argument("--inter_state_file", type=str)
    parser.add_argument("--step_ratio_diff", type=float, default=0.4)
    parser.add_argument("--negative_threshold", type=float, default=1.0)
    # Temporarily disable this option because I don't know how to enable it and process the full examples at the same time.
    # parser.add_argument("--early_stop", action="store_true", default=False)
    parser.add_argument("--step_id_clip", type=str, default="(6,30)")
    parser.add_argument("--negative_sample_num", type=int, default=2)
    # parser.add_argument("--exclude_full", action="store_true", default=False)
    parser.add_argument("--full_sampling", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

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

    step_id_clip = eval(args.step_id_clip)

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
            if x["step_id"] < step_id_clip[0] or x["step_id"] > step_id_clip[1]:
                continue
            resp_id2states[x["resp_id"]].append(x)

        for resp_id, states in resp_id2states.items():
            resp_id2states[resp_id] = sorted(states, key=lambda tmp: tmp["step_id"])

        full_rationales = []
        partial_rationales = []
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

                partial_rationales.append(state)

                # if error and args.early_stop:
                #     break

            # if error and args.early_stop:
            #     continue

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

            full_rationales.append(resp_full_state)

        # value2rationales = collections.defaultdict(list)
        # for r in all_rationales:
        #     value2rationales[r["value"]].append(r)
        values = set()
        value2full_rationales = collections.defaultdict(list)
        for r in full_rationales:
            value2full_rationales[r["value"]].append(r)
            values.add(r["value"])
        value2partial_rationales = collections.defaultdict(list)
        for r in partial_rationales:
            value2partial_rationales[r["value"]].append(r)
            values.add(r["value"])

        # values = list(value2rationales.keys())
        # values.sort(reverse=True)
        values = list(values)
        values.sort(reverse=True)
        for chosen_v in values:
            if chosen_v - args.diff < 0:
                break

            # Full-full comparison
            for reject_v in range(0, int(chosen_v - args.diff + 1)):
                if reject_v not in value2full_rationales:
                    continue

                res = iterate_comparison_pair(idx,
                                              value2full_rationales[chosen_v],
                                              value2full_rationales[reject_v],
                                              args.step_ratio_diff,
                                              args.negative_sample_num if args.full_sampling else 100000)
                outputs.extend(res[0])
                statis = res[1]
                num_partial_longer_win += statis["num_partial_longer_win"]
                num_partial_shorter_win += statis["num_partial_shorter_win"]
                num_cross_full_win += statis["num_cross_full_win"]
                num_cross_part_win += statis["num_cross_part_win"]

            # Full-partial comparison
            for reject_v in range(0, int(chosen_v - args.diff + 1)):
                if reject_v not in value2partial_rationales:
                    continue

                res = iterate_comparison_pair(idx,
                                              value2full_rationales[chosen_v],
                                              value2partial_rationales[reject_v],
                                              args.step_ratio_diff,
                                              args.negative_sample_num)
                outputs.extend(res[0])
                statis = res[1]
                num_partial_longer_win += statis["num_partial_longer_win"]
                num_partial_shorter_win += statis["num_partial_shorter_win"]
                num_cross_full_win += statis["num_cross_full_win"]
                num_cross_part_win += statis["num_cross_part_win"]

            # Partial-full comparison
            for reject_v in range(0, int(chosen_v - args.diff + 1)):
                if reject_v not in value2full_rationales:
                    continue

                res = iterate_comparison_pair(idx,
                                              value2partial_rationales[chosen_v],
                                              value2full_rationales[reject_v],
                                              args.step_ratio_diff,
                                              args.negative_sample_num)
                outputs.extend(res[0])
                statis = res[1]
                num_partial_longer_win += statis["num_partial_longer_win"]
                num_partial_shorter_win += statis["num_partial_shorter_win"]
                num_cross_full_win += statis["num_cross_full_win"]
                num_cross_part_win += statis["num_cross_part_win"]

            # Partial-partial comparison
            for reject_v in range(0, int(chosen_v - args.diff + 1)):
                if reject_v not in value2partial_rationales:
                    continue

                res = iterate_comparison_pair(idx,
                                              value2partial_rationales[chosen_v],
                                              value2partial_rationales[reject_v],
                                              args.step_ratio_diff,
                                              args.negative_sample_num)
                outputs.extend(res[0])
                statis = res[1]
                num_partial_longer_win += statis["num_partial_longer_win"]
                num_partial_shorter_win += statis["num_partial_shorter_win"]
                num_cross_full_win += statis["num_cross_full_win"]
                num_cross_part_win += statis["num_cross_part_win"]

    full_pairs = 0
    for x in outputs:
        if x["chosen_full"] and x["reject_full"]:
            full_pairs += 1

    print(f"Jumped: {jumped}")
    print(len(outputs))
    print(f"Masked: {masked}")
    print(f"Full pairs: {full_pairs}")
    print(f"Partial longer win: {num_partial_longer_win}")
    print(f"Partial shorter win: {num_partial_shorter_win}")
    print(f"Cross full win: {num_cross_full_win}")
    print(f"Cross partial win: {num_cross_part_win}")

    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
