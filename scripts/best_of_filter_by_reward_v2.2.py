import argparse
import json
import os
import re
from glob import glob
from typing import Dict

import torch

"""
Updated from Ver.1.2

Ver.2.2:
    - Recompute the reward by only use the probability of `label==3`.
    - Add label positions.

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
    for resp_id, resp in enumerate(item["response"]):
        if resp not in response2reward:
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

    return chosen_responses, responses, reject_responses, correct_num


def best_of_n_filter_inter_states(item, best_of: int, response2reward: Dict[str, float]):
    responses = []
    for state_id, state in enumerate(item["inter_states"]):
        resp = state["state"]
        if resp not in response2reward:
            continue
        responses.append((state_id, response2reward[resp]))

    responses = sorted(responses, key=lambda x: x[1], reverse=True)
    chosen_responses = responses[:best_of]
    reject_responses = responses[best_of:]
    chosen_responses = sorted(chosen_responses, key=lambda x: x[0])
    reject_responses = sorted(reject_responses, key=lambda x: x[0])
    chosen_responses = [item["inter_states"][state_id]["state"] for state_id, _ in chosen_responses]
    reject_responses = [item["inter_states"][state_id]["state"] for state_id, _ in reject_responses]
    return chosen_responses, reject_responses


# def logit2prob(logits):
#     probs = torch.softmax(logits, dim=-1)
#     return probs[:, 3]
def logit2prob(logits, prob_labels=(3,)):
    if prob_labels:
        probs = torch.softmax(logits, dim=-1)
        # Sum the probabilities along the `prob_labels`.
        return probs[:, prob_labels].sum(dim=-1)

    return logits[:, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--max_neg_num", type=int, default=100)
    parser.add_argument("--pos_margin", type=float, default=2.0)
    parser.add_argument("--reduction", type=str, default="product", choices=["product", "min", "sum"])
    parser.add_argument("--prob_labels", type=str, default="(3,)", help="The labels to compute the probability.")
    parser.add_argument("--up_sampling", type=int, default=1)
    parser.add_argument("--min_pos_step", type=int, default=-1)
    parser.add_argument("--min_neg_step", type=int, default=-1)
    parser.add_argument("--exclude_file", type=str, default=None)
    args = parser.parse_args()

    args.prob_labels = eval(args.prob_labels)
    print(args.prob_labels)
    print(args.reduction)

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    data = []
    for file in files:
        data += json.load(open(file, "r"))

    rewards = json.load(open(args.reward_file, "r"))
    response2reward = {}
    response2full_rewards = {}
    duplicates = set()
    cnt = 0
    for item in rewards:
        if item["response"] in response2reward:
            # print("duplicate response")
            duplicates.add(item["response"])
            cnt += 1

        logits = torch.tensor(item["ending_logits"])
        probs = logit2prob(logits, prob_labels=args.prob_labels)
        response2full_rewards[item["response"]] = probs.tolist()
        if args.reduction == "product":
            response2reward[item["response"]] = probs.prod().item()
        elif args.reduction == "min":
            response2reward[item["response"]] = probs.min().item()
        elif args.reduction == "sum":
            response2reward[item["response"]] = probs.sum().item()
        elif args.reduction == "last":
            response2reward[item["response"]] = probs[-1].item()
        else:
            raise ValueError(f"Unsupported reduction: {args.reduction}")

    exclude_cnt = 0
    exclude_ids = set()
    if args.exclude_file is not None:
        exclude_data = json.load(open(args.exclude_file, "r"))
        for item in exclude_data:
            exclude_ids.add(item["id"])
            exclude_cnt += 1

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)
    print("exclude items", exclude_cnt)

    filtered = []
    pos2pos = []
    reduced = 0
    pos_pair = 0
    for item in data:
        chosen_responses, pos_response_rewards, reject_responses, correct_num = best_of_n_filter(item, args.best_of, response2reward)
        for chosen in chosen_responses:
            for reject in reject_responses[:args.max_neg_num]:
                filtered.append({
                    "chosen": chosen,
                    "reject": reject,
                    "id": item["id"],
                    "is_full": True,
                    "chosen_full_rewards": response2full_rewards[chosen],
                    "reject_full_rewards": response2full_rewards[reject],
                    "chosen_reward": response2reward[chosen],
                    "reject_reward": response2reward[reject],
                })
            if args.min_pos_step > 0 and len(chosen.split("\n")) < args.min_pos_step:
                continue
            if item["id"] in exclude_ids:
                continue
            for pos_id, pos_reward in pos_response_rewards:
                if args.min_neg_step > 0 and len(item["response"][pos_id].split("\n")) < args.min_neg_step:
                    continue
                if response2reward[chosen] - pos_reward > args.pos_margin:
                    pos2pos.append({
                        "chosen": chosen,
                        "reject": item["response"][pos_id],
                        "id": item["id"],
                        "is_full": True,
                        "chosen_full_rewards": response2full_rewards[chosen],
                        "reject_full_rewards": response2full_rewards[item["response"][pos_id]],
                        "chosen_reward": response2reward[chosen],
                        "reject_reward": response2reward[item["response"][pos_id]],
                    })
                    pos_pair += 1
        if correct_num < args.best_of:
            reduced += 1

    print("Positive pairs", len(pos2pos))
    pos2pos = pos2pos * args.up_sampling
    filtered += pos2pos

    print("Reduced", reduced)
    # print("Positive pairs", pos_pair)
    print("Positive pairs by up-sampling:", len(pos2pos))
    print(f"Candidates: {len(data)}")
    print("Collected amount of samples with rewards", len(filtered))
    print(f"Save to {args.output_file}")
    json.dump(filtered, open(args.output_file, "w"), indent=4)


if __name__ == '__main__':
    main()
