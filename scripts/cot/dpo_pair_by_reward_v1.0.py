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


def best_of_n_filter(item, best_of: int, id2reward: Dict[str, float]):
    incorrect_responses = []
    responses = []
    assert len(item["response"]) == len(item["pred"])
    for resp_id, (resp, pred) in enumerate(zip(item["response"], item["pred"])):
        idx = f"{item['id']}_{resp_id}"
        if idx not in id2reward:
            print(f"Warning: Not a valid response: {resp}")
            continue
        if isinstance(item["label"], int):
            if pred and ord(pred) - ord("A") == item["label"]:
                v = 1
            else:
                v = 0
        elif isinstance(item["label"], str):
            if pred == item["label"]:
                v = 1
            else:
                v = 0
        else:
            raise ValueError(f"Unknown type of label: {type(item['label'])}")

        if v == 0:
            incorrect_responses.append((resp_id, id2reward[idx], resp))
        else:
            responses.append((resp_id, id2reward[idx], resp))

    correct_num = len(responses)

    responses = sorted(responses, key=lambda x: x[1], reverse=True)
    chosen_responses = responses[:best_of]
    reject_responses = incorrect_responses
    # chosen_responses = sorted(chosen_responses, key=lambda x: x[0])
    reject_responses = sorted(reject_responses, key=lambda x: x[1], reverse=True)
    # chosen_responses = [item["response"][resp_id] for resp_id, _ in chosen_responses]
    # reject_responses = [item["response"][resp_id] for resp_id, _ in reject_responses]

    return chosen_responses, responses, reject_responses, correct_num


def logit2prob(logits, prob_labels=(3,)):
    probs = torch.softmax(logits, dim=-1)
    # Sum the probabilities along the `prob_labels`.
    return probs[:, prob_labels].sum(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--max_neg_num", type=int, default=100)
    parser.add_argument("--pos_margin", type=float, default=2.0)
    parser.add_argument("--reduction", type=str, default="product", choices=["product", "min"])
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
        files = glob(args.input_file, recursive=True)
    print(files)
    data = []
    for file in files:
        data += json.load(open(file, "r"))

    if os.path.exists(args.reward_file):
        files = [args.reward_file]
    else:
        files = glob(args.reward_file, recursive=True)
    rewards = []
    for file in files:
        rewards += json.load(open(file, "r"))
    print(files)
    id2reward = {}
    id2full_reward = {}
    cnt = 0
    for item in rewards:
        logits = torch.tensor(item["ending_logits"])
        probs = logit2prob(logits, prob_labels=args.prob_labels)
        id2full_reward[item["index"]] = probs.tolist()
        if args.reduction == "product":
            id2reward[item["index"]] = probs.prod().item()
        elif args.reduction == "min":
            id2reward[item["index"]] = probs.min().item()
        elif args.reduction == "sum":
            id2reward[item["index"]] = probs.sum().item()
        elif args.reduction == "last":
            id2reward[item["index"]] = probs[-1].item()
        else:
            raise ValueError(f"Unsupported reduction: {args.reduction}")

    exclude_cnt = 0
    exclude_ids = set()
    if args.exclude_file is not None:
        exclude_data = json.load(open(args.exclude_file, "r"))
        for item in exclude_data:
            exclude_ids.add(item["id"])
            exclude_cnt += 1

    print("collected rewards", len(id2reward))
    print("duplicate responses", cnt)
    print("exclude items", exclude_cnt)

    filtered = []
    pos2pos = []
    reduced = 0
    pos_pair = 0
    for item in data:
        chosen_responses, pos_response_rewards, reject_responses, correct_num = best_of_n_filter(item, args.best_of, id2reward)
        for chosen in chosen_responses:
            for reject in reject_responses[:args.max_neg_num]:
                filtered.append({
                    "chosen": chosen[2],
                    "reject": reject[2],
                    "id": item["id"],
                    "is_full": True,
                    "chosen_full_rewards": id2full_reward[f"{item['id']}_{chosen[0]}"],
                    "reject_full_rewards": id2full_reward[f"{item['id']}_{reject[0]}"],
                    "chosen_reward": chosen[1],
                    "reject_reward": reject[1],
                })
            if args.min_pos_step > 0 and len(chosen[2].split("\n")) < args.min_pos_step:
                continue
            if item["id"] in exclude_ids:
                continue
            for pos_id, pos_reward, pos_resp in pos_response_rewards:
                if args.min_neg_step > 0 and len(item["response"][pos_id].split("\n")) < args.min_neg_step:
                    continue
                if chosen[1] - pos_reward > args.pos_margin:
                    pos2pos.append({
                        "chosen": chosen[2],
                        "reject": item["response"][pos_id],
                        "id": item["id"],
                        "is_full": True,
                        "chosen_full_rewards": id2full_reward[f"{item['id']}_{chosen[0]}"],
                        "reject_full_rewards": id2full_reward[f"{item['id']}_{pos_id}"],
                        "chosen_reward": chosen[1],
                        "reject_reward": pos_reward,
                    })
                    pos_pair += 1
        if correct_num < args.best_of:
            reduced += 1

    print("Positive pairs", len(pos2pos))
    pos2pos = pos2pos * args.up_sampling
    filtered += pos2pos

    print("Reduced", reduced)
    print("Positive pairs by up-sampling:", len(pos2pos))
    print(f"Candidates: {len(data)}")
    print("Collected amount of samples with rewards", len(filtered))
    print(f"Save to {args.output_file}")
    json.dump(filtered, open(args.output_file, "w"), indent=4)
    json.dump(pos2pos, open(args.output_file.replace(".json", "_pos2pos.json"), "w"), indent=4)


if __name__ == '__main__':
    main()
