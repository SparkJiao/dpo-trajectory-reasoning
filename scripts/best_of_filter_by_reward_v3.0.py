import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List
import numpy as np

import torch

"""
Updated from Ver.1.2

Ver.2.2:
    - Recompute the reward by only use the probability of `label==3`.
    - Add label positions.
    
Ver.3.0:
    - Use rejection sampling instead of fixed margin.

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


def logit2prob(logits, prob_labels=(3,)):
    probs = torch.softmax(logits, dim=-1)
    # Sum the probabilities along the `prob_labels`.
    return probs[:, prob_labels].sum(dim=-1)


def conduct_rejection_sampling(response_candidates: List[str], response_rewards: List[float], num_samples: int, beta: float):
    """Conducts rejection sampling guided by rewards. 
    
    Args: 
        response_candidates: response candidates from sft policy 
        response_rewards: response rewards. 
        num_samples: number of samples to sub-sample. 
        beta: beta parameter in KL-constrained reward maximization objective. 
    Returns: 
        Rejection sampled sequences from the optimal policy.
    """
    candidates = {c: r for c, r in zip(response_candidates, response_rewards)}
    accepted = []
    while len(accepted) < num_samples:
        max_reward = max(candidates.values())
        to_remove = []
        for c, r in candidates.items():
            u = np.random.uniform()
            if u >= np.exp((r - max_reward) / beta):
                continue
            accepted.append(c)
            to_remove.append(c)
            if len(accepted) == num_samples:
                break
        for c in to_remove:
            candidates.pop(c)

    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--max_neg_num", type=int, default=100)
    parser.add_argument("--accept_ratio", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
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
            # print("duplicate response")
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
    for item in data:
        chosen_responses, pos_response_rewards, reject_responses, correct_num = best_of_n_filter(item, args.best_of, response2reward)
        for chosen in chosen_responses:
            for reject in reject_responses[:args.max_neg_num]:
                filtered.append({
                    "chosen": chosen,
                    "reject": reject,
                    "id": item["id"],
                    "is_full": True,
                })

        num_samples = int(args.accept_ratio * len(pos_response_rewards))
        if num_samples == 0:
            continue
        accepted_ids = conduct_rejection_sampling(
            response_candidates=[r for r, _ in pos_response_rewards],
            response_rewards=[r for _, r in pos_response_rewards],
            num_samples=num_samples,
            beta=args.beta,
        )
        rejected_ids = set([pos_id for pos_id, _ in pos_response_rewards]) - set(accepted_ids)
        for chosen_id in accepted_ids:
            for reject_id in rejected_ids:
                pos2pos.append({
                    "chosen": item["response"][chosen_id],
                    "reject": item["response"][reject_id],
                    "id": item["id"],
                    "is_full": True,
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
    print("Averaged accepted responses:", pos_pair / len(data))
    print(f"Candidates: {len(data)}")
    print("Collected amount of samples with rewards", len(filtered))
    print(f"Save to {args.output_file}")
    json.dump(filtered, open(args.output_file, "w"), indent=4)


if __name__ == '__main__':
    main()
