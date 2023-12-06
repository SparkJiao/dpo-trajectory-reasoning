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
    # parser.add_argument("--decay", type=float, default=0.9)
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

        item_states = item["inter_states"]
        resp_id2states = collections.defaultdict(list)
        for x_id, x in enumerate(item_states):
            if x["state"] not in response2reward:
                jumped += 1
                continue
            resp_id2states[x["resp_id"]].append((x, x_id))

        for x_i, x in enumerate(item_states):
            if x["state"] not in response2reward:
                continue
            step_id_x = x["step_id"]
            val_x = response2reward[x["state"]]
            for y_i, y in enumerate(item_states):
                if x_i == y_i:
                    continue
                if y["state"] not in response2reward:
                    continue
                step_id_y = y["step_id"]
                val_y = response2reward[y["state"]]
                # if step_id_x < step_id_y:
                #     val_x = val_x * args.decay ** (step_id_y - step_id_x)
                # elif step_id_x > step_id_y:
                #     val_y = val_y * args.decay ** (step_id_x - step_id_y)
                if val_x - val_y >= args.diff:
                    if step_id_x < step_id_y:
                        num_partial_shorter_win += 1
                    elif step_id_x > step_id_y:
                        num_partial_longer_win += 1

                    outputs.append({
                        "id": idx,
                        "chosen": x["state"],
                        "reject": y["state"],
                        "is_full": False,
                        "val_diff": val_x - val_y,
                    })

        for x_i, x in enumerate(item_states):
            # if x_i not in state_id2values[idx]:
            #     continue
            if x["state"] not in response2reward:
                continue
            step_id_x = x["step_id"]
            # val_x, res_x = state_id2values[idx][x_i]
            val_x = response2reward[x["state"]]
            for r_id, resp in enumerate(item["response"]):
                # r_steps = process_response(resp)
                # step_id_r = len(r_steps) - 1
                # val_r = resp["value"]
                if resp not in response2reward:
                    continue
                val_r = response2reward[resp]
                # if step_id_x < step_id_r:
                #     val_x = val_x * args.decay ** (step_id_r - step_id_x)
                # elif step_id_x > step_id_r:
                #     val_r = val_r * args.decay ** (step_id_x - step_id_r)
                if val_x - val_r >= args.diff:
                    num_cross_part_win += 1
                    outputs.append({
                        "id": idx,
                        "chosen": x["state"],
                        "reject": resp,
                        "chosen_full": False,
                        "reject_full": True,
                        "val_diff": val_x - val_r,
                    })
                elif val_r - val_x >= args.diff:
                    final_value = parse_leaf_node_value(resp, item["label"])
                    if final_value == 0:
                        continue

                    num_cross_full_win += 1
                    outputs.append({
                        "id": idx,
                        "chosen": resp,
                        "reject": x["state"],
                        "chosen_full": True,
                        "reject_full": False,
                        "val_diff": val_r - val_x,
                    })

    print(f"Jumped: {jumped}")
    print(len(outputs))
    print("Amount of partial longer win: ", num_partial_longer_win)
    print("Amount of partial shorter win: ", num_partial_shorter_win)
    print("Amount of cross full win: ", num_cross_full_win)
    print("Amount of cross part win: ", num_cross_part_win)
    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
