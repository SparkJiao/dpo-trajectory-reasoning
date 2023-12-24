import argparse
import json
import os
import re
from glob import glob
from typing import Dict
import torch

"""
Updated from Ver.1.1

Ver.1.3:
    - Recollect the step wise rewards for each response.
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
    elif len(preds) > 1:
        return 0
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            return 0


def best_of_n_filter(item, best_of: int, response2reward: Dict[str, float]):
    incorrect_responses = []
    responses = []
    missed_responses = 0
    for resp_id, resp in enumerate(item["response"]):
        if resp not in response2reward:
            missed_responses += 1
            continue
        v = parse_leaf_node_value(resp, item["label"])
        if v == 0:
            incorrect_responses.append((resp_id, response2reward[resp]))
        else:
            responses.append((resp_id, response2reward[resp]))

    correct_num = len(responses)

    responses = sorted(responses, key=lambda x: x[1], reverse=True)
    chosen_responses = responses[:best_of]
    reject_responses = incorrect_responses
    chosen_responses = sorted(chosen_responses, key=lambda x: x[0])
    reject_responses = sorted(reject_responses, key=lambda x: x[1], reverse=True)
    chosen_responses = [item["response"][resp_id] for resp_id, _ in chosen_responses]
    reject_responses = [item["response"][resp_id] for resp_id, _ in reject_responses]
    return chosen_responses, reject_responses, correct_num, missed_responses


def best_of_n_filter_inter_states(item, best_of: int, response2reward: Dict[str, float]):
    responses = []
    missed_responses = 0
    for state_id, state in enumerate(item["inter_states"]):
        resp = state["state"]
        if resp not in response2reward:
            missed_responses += 1
            continue
        responses.append((state_id, response2reward[resp]))

    responses = sorted(responses, key=lambda x: x[1], reverse=True)
    chosen_responses = responses[:best_of]
    reject_responses = responses[best_of:]
    chosen_responses = sorted(chosen_responses, key=lambda x: x[0])
    reject_responses = sorted(reject_responses, key=lambda x: x[1], reverse=True)
    chosen_responses = [item["inter_states"][state_id]["state"] for state_id, _ in chosen_responses]
    reject_responses = [item["inter_states"][state_id]["state"] for state_id, _ in reject_responses]
    return chosen_responses, reject_responses, missed_responses


def process_response(response: str):
    lines = response.split("\n")
    lines = list(filter(lambda x: x[1].startswith("Thought ") or x[1].startswith("Action ") or x[1].startswith("Observation "), enumerate(lines)))
    return lines


def logit2prob(logits):
    probs = torch.softmax(logits, dim=-1)
    probs = probs[:, 2] + probs[:, 3]
    return probs


def test(response: str, max_seq_length: int):
    steps = process_response(response)
    raw_lines = response.split("\n")
    endings = []
    resp_start = False
    for step_id, step in steps:
        if not resp_start:
            if step.startswith("Thought 1:"):
                resp_start = True
            else:
                continue
        # partial_traj = "\n".join(raw_lines[:(step_id + 1)])
        endings.append(0)

    assert resp_start, response
    assert len(endings) > 0, (response, steps)
    assert len(endings) == len(steps), (response, steps)
    return endings


def collect_step_reward(reward_item, reduction="product"):
    ending_logits = torch.tensor(reward_item["ending_logits"])
    probs = logit2prob(ending_logits)
    if reduction == "product":
        acc_reward = 1.0
    elif reduction == "sum":
        acc_reward = 0.0
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # if "Thought 1:" not in reward_item["response"]:
    if not reward_item["response"].startswith("Thought 1:"):
        response = "Thought 1: " + reward_item["response"]
    else:
        response = reward_item["response"]

    raw_lines = reward_item["response"].split("\n")
    raw_steps = process_response(response)

    step_rewards = []
    for step_id, step in enumerate(probs):
        step_reward = step.item()
        if reduction == "product":
            acc_reward *= step_reward
        elif reduction == "sum":
            acc_reward += step_reward
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        tmp = test("Thought 1: " + reward_item["response"], 4096)

        try:
            partial_traj = "\n".join(raw_lines[:(raw_steps[step_id][0] + 1)])
        except IndexError:
            print(f"Warning: {probs.shape}, {len(tmp)}, {step_id}, {len(raw_steps)}, {len(raw_lines)}")
            print(raw_steps)
            print(raw_lines)
            print(len(process_response(reward_item["input"])))
            print(process_response(reward_item["input"]))
            raise IndexError

        step_rewards.append((partial_traj, acc_reward))
    return step_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--max_neg_num", type=int, default=100)
    parser.add_argument("--inter_best_of", type=int, default=1)
    parser.add_argument("--inter_max_neg_num", type=int, default=100)
    parser.add_argument("--inter_margin", type=float, default=0.5)
    parser.add_argument("--reduction", type=str, default="product")
    parser.add_argument("--contrastive", action="store_true", default=False)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    data = []
    for file in files:
        data += json.load(open(file, "r"))

    rewards = json.load(open(args.reward_file, "r"))
    response2reward = {}
    duplicates = set()
    cnt = 0
    for item in rewards:
        if item["response"] in response2reward:
            # print("duplicate response")
            duplicates.add(item["response"])
            cnt += 1
        if isinstance(item["reward"], list):
            assert len(item["reward"]) == 1
            response2reward[item["response"]] = item["reward"][0]
        elif isinstance(item["reward"], float):
            response2reward[item["response"]] = item["reward"]
        else:
            raise ValueError(f"Unsupported type of reward: {item['reward'].__class__}")

        step_rewards = collect_step_reward(item, reduction=args.reduction)
        for partial_traj, acc_reward in step_rewards:
            response2reward[partial_traj] = acc_reward

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)

    filtered = []
    reduced = 0
    inter_num = 0
    missed_full_num = 0
    missed_partial_num = 0
    for item in data:
        chosen_responses, reject_responses, correct_num, missed_num = best_of_n_filter(item, args.best_of, response2reward)
        missed_full_num += missed_num
        for chosen in chosen_responses:
            for reject in reject_responses[:args.max_neg_num]:
                filtered.append({
                    "chosen": chosen,
                    "reject": reject,
                    "id": item["id"],
                    "is_full": True,
                })
        if correct_num < args.best_of:
            reduced += 1

        if args.inter_best_of > 0:
            chosen_responses, reject_responses, missed_num = best_of_n_filter_inter_states(item, args.inter_best_of, response2reward)
            missed_partial_num += missed_num
            for chosen in chosen_responses:
                tmp = 0
                for reject in reject_responses:
                    if tmp >= args.inter_max_neg_num:
                        break
                    if response2reward[chosen] - response2reward[reject] > args.inter_margin:
                        filtered.append({
                            "chosen": chosen,
                            "reject": reject,
                            "id": item["id"],
                            "is_full": False,
                        })
                        inter_num += 1
                        tmp += 1

    print("Reduced", reduced)
    print(f"Candidates: {len(data)}")
    print(f"Inter: {inter_num}")
    print(f"Missed full: {missed_full_num}")
    print(f"Missed partial: {missed_partial_num}")
    print("Collected amount of samples with rewards", len(filtered))
    print(f"Save to {args.output_file}")
    json.dump(filtered, open(args.output_file, "w"), indent=4)


if __name__ == '__main__':
    main()
