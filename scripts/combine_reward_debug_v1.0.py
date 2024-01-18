import argparse
import collections
import json
import os
import re
from glob import glob
from typing import Dict
import torch


def logit2prob(logits, prob_labels=(3,)):
    probs = torch.softmax(logits, dim=-1)
    # Sum the probabilities along the `prob_labels`.
    if len(probs.shape) == 1:
        return probs[prob_labels].sum()
    return probs[:, prob_labels].sum(dim=-1)


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


def parse_prediction(response: str):
    groups = response.split("Finish")
    if len(groups) < 2:
        # print(f"Warning: Not a valid response: {response}")
        return None
    response = groups[1]
    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        return None
    elif len(preds) > 1:
        return None
    else:
        return ord(preds[0]) - ord("A")


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
    return chosen_responses, reject_responses, correct_num


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--reduction", type=str, default="product")
    parser.add_argument("--remove_last", default=False, action="store_true")
    parser.add_argument("--prob_labels", type=str, default="(3,)", help="The labels to compute the probability.")
    parser.add_argument("--remove_action", default=False, action="store_true")
    parser.add_argument("--orm", default=False, action="store_true")
    args = parser.parse_args()
    print(args.reduction)
    print(args.prob_labels)

    args.prob_labels = eval(args.prob_labels)

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
    reduced = 0
    for item in rewards:
        if item["response"] in response2reward:
            duplicates.add(item["response"])
            cnt += 1
        if args.orm:
            logits = torch.tensor(item["logits"])
        else:
            logits = torch.tensor(item["ending_logits"])
        probs = logit2prob(logits, prob_labels=args.prob_labels)

        if args.remove_action:
            step_types = item["step_types"]
            # Remove all step rewards of "Action".
            assert len(probs) == len(step_types)
            reduced += 1
            probs = torch.tensor([prob.item() for prob, step_type in zip(probs, step_types) if step_type != "Action"])

        if args.remove_last:
            probs = probs[:-1]
        if args.orm:
            response2reward[item["response"]] = probs.item()
        elif args.reduction == "product":
            response2reward[item["response"]] = probs.prod().item()
        elif args.reduction == "sum":
            response2reward[item["response"]] = probs.sum().item()
        elif args.reduction == "last":
            response2reward[item["response"]] = probs[-1].item()
        elif args.reduction == "min":
            response2reward[item["response"]] = probs.min().item()
        else:
            raise ValueError(f"Unsupported reduction: {args.reduction}")

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)

    sc_correct = 0
    rm_correct = 0
    for item in data:
        cleaned_responses = []
        preds = collections.Counter()
        rewards = []
        for resp_id, resp in enumerate(item["response"]):
            if resp not in response2reward:
                continue
            v = parse_leaf_node_value(resp, item["label"])
            if v == 0:
                cleaned_responses.append({
                    "response": resp,
                    "reward": response2reward[resp],
                    "is_correct": False,
                })
            else:
                cleaned_responses.append({
                    "response": resp,
                    "reward": response2reward[resp],
                    "is_correct": True,
                })

            pred = parse_prediction(resp)
            if pred is not None:
                preds[pred] += 1
                rewards.append((pred, response2reward[resp]))
        item["response_reward"] = cleaned_responses

        if len(preds) > 0:
            sc_correct += int(preds.most_common(1)[0][0] == item["label"])

        if len(rewards) > 0:
            rewards = sorted(rewards, key=lambda x: x[1], reverse=True)
            rm_correct += int(rewards[0][0] == item["label"])

    json.dump(data, open(args.output_file, "w"), indent=2)
    print("sc acc", sc_correct / len(data))
    print("rm acc", rm_correct / len(data))
    print("reduced", reduced)


if __name__ == '__main__':
    main()
