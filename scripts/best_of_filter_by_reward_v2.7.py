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
    
Ver.2.7:
    - If one sample serves as the negative candidate in some pairs, we remove all the samples where it serves as the positive candidate.

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


def best_of_n_filter(item, response2reward: Dict[str, float]):
    incorrect_responses = []
    responses = []
    for resp_id, resp in enumerate(item["response"]):
        v = parse_leaf_node_value(resp, item["label"])
        if v == 0:
            incorrect_responses.append((resp_id, response2reward[resp]))
        else:
            if resp not in response2reward:
                continue
            responses.append((resp_id, response2reward[resp]))

    correct_num = len(responses)

    return responses, incorrect_responses, correct_num


def logit2prob(logits, prob_labels=(3,)):
    probs = torch.softmax(logits, dim=-1)
    # Sum the probabilities along the `prob_labels`.
    return probs[:, prob_labels].sum(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pos_margin", type=float, default=2.0)
    parser.add_argument("--reduction", type=str, default="product", choices=["product", "min"])
    parser.add_argument("--prob_labels", type=str, default="(3,)", help="The labels to compute the probability.")
    parser.add_argument("--up_sampling", type=int, default=1)
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
    duplicates = set()
    cnt = 0
    for item in rewards:
        if item["response"] in response2reward:
            duplicates.add(item["response"])
            cnt += 1

        logits = torch.tensor(item["ending_logits"])
        probs = logit2prob(logits, prob_labels=args.prob_labels)
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

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)

    filtered = []
    pos2pos = []
    reduced = 0
    pos_pair = 0
    neg_pair = 0
    for item in data:
        chosen_responses, reject_responses, correct_num = best_of_n_filter(item, response2reward)

        surpassed_ids = set()
        for chosen_id_x, reward_x in chosen_responses:
            for chosen_id_y, reward_y in chosen_responses:
                if chosen_id_x == chosen_id_y:
                    continue
                if reward_x - reward_y > args.pos_margin:
                    surpassed_ids.add(chosen_id_y)
                    pos2pos.append({
                        "chosen": item["response"][chosen_id_x],
                        "reject": item["response"][chosen_id_y],
                        "id": item["id"],
                        "is_full": False,
                    })
                    pos_pair += 1

        for chosen_id, reward in chosen_responses:
            if chosen_id in surpassed_ids:
                continue
            for reject_id, reject_reward in reject_responses:
                filtered.append({
                    "chosen": item["response"][chosen_id],
                    "reject": item["response"][reject_id],
                    "id": item["id"],
                    "is_full": False,
                })
                neg_pair += 1

        # if correct_num < args.best_of:
        #     reduced += 1

    print("Positive pairs", len(pos2pos))
    pos2pos = pos2pos * args.up_sampling
    filtered += pos2pos

    # print("Reduced", reduced)
    print("Negative pairs", neg_pair)
    print("Positive pairs by up-sampling:", len(pos2pos))
    print(f"Candidates: {len(data)}")
    print("Collected amount of samples with rewards", len(filtered))
    print(f"Save to {args.output_file}")
    json.dump(filtered, open(args.output_file, "w"), indent=4)


if __name__ == '__main__':
    main()
