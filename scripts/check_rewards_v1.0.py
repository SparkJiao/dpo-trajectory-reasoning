import argparse
import collections
import json
import os
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


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


def logit2prob(logits, prob_labels=(3,)):
    probs = torch.softmax(logits, dim=-1)
    # Sum the probabilities along the `prob_labels`.
    return probs[:, prob_labels].sum(dim=-1)


def plot_bar(x, y, xlabel="Value", ylabel="Frequency", title="Histogram"):
    plt.figure(figsize=(16, 8))
    plt.bar(x, y)
    plt.xlabel(xlabel, fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.tick_params(axis='x', labelsize=28)
    plt.tick_params(axis='y', labelsize=28)
    plt.title(title, fontsize=36)
    plt.grid(True)
    plt.show()
    plt.savefig(f"{'_'.join(title.split(' '))}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--reward_file", type=str, required=True)
    parser.add_argument("--correct_only", action="store_true", default=False)
    parser.add_argument("--step_cutoff", type=int, default=10000)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    responses = []
    for file in files:
        responses += json.load(open(file, "r"))

    resp2v = {}
    for item in responses:
        for resp in item["response"]:
            if resp not in resp2v:
                resp2v[resp] = parse_leaf_node_value(resp, item["label"])

    rewards = json.load(open(args.reward_file, "r"))

    res_raw_sum = collections.defaultdict(list)
    res_prob_sum = collections.defaultdict(list)
    res_prob_prod = collections.defaultdict(list)
    step_raw = collections.defaultdict(list)
    step_prob = collections.defaultdict(list)
    response_set = set()
    for item in tqdm(rewards, total=len(rewards)):
        if item["response"] in response_set:
            continue

        if args.correct_only:
            if item["response"] not in resp2v:
                continue
            if resp2v[item["response"]] == 0:
                continue

        response_set.add(item["response"])

        logits = torch.tensor(item["ending_logits"])
        probs = logit2prob(logits, prob_labels=(2, 3))
        raw_sum = 0
        prob_sum = 0
        prob_prod = 1
        for step_id in range(len(probs)):
            if step_id > args.step_cutoff:
                break

            raw_sum += logits[step_id][3].item()
            prob_sum += probs[step_id].item()
            prob_prod *= probs[step_id].item()

            res_raw_sum[step_id].append(raw_sum)
            res_prob_sum[step_id].append(prob_sum)
            res_prob_prod[step_id].append(prob_prod)

            step_raw[step_id].append(logits[step_id][3].item())
            step_prob[step_id].append(probs[step_id].item())

    for step_id in range(len(res_raw_sum)):
        res_raw_sum[step_id] = np.mean(res_raw_sum[step_id])
        res_prob_sum[step_id] = np.mean(res_prob_sum[step_id])
        res_prob_prod[step_id] = np.mean(res_prob_prod[step_id])

        step_raw[step_id] = np.mean(step_raw[step_id])
        step_prob[step_id] = np.mean(step_prob[step_id])

    # sort according to the step_id
    res_raw_sum = collections.OrderedDict(sorted(res_raw_sum.items(), key=lambda x: x[0]))
    res_prob_sum = collections.OrderedDict(sorted(res_prob_sum.items(), key=lambda x: x[0]))
    res_prob_prod = collections.OrderedDict(sorted(res_prob_prod.items(), key=lambda x: x[0]))

    step_raw = collections.OrderedDict(sorted(step_raw.items(), key=lambda x: x[0]))
    step_prob = collections.OrderedDict(sorted(step_prob.items(), key=lambda x: x[0]))

    print(res_raw_sum)
    print(res_prob_sum)
    print(res_prob_prod)

    plot_bar(list(res_raw_sum.keys()), list(res_raw_sum.values()), xlabel="Step ID", ylabel="Average Raw Sum",
             title=f"Average Raw Sum{' Correct' if args.correct_only else ''}")
    plot_bar(list(res_prob_sum.keys()), list(res_prob_sum.values()), xlabel="Step ID", ylabel="Average Prob Sum",
             title=f"Average Prob Sum{' Correct' if args.correct_only else ''}")
    plot_bar(list(res_prob_prod.keys()), list(res_prob_prod.values()), xlabel="Step ID", ylabel="Average Prob Prod",
             title=f"Average Prob Prod{' Correct' if args.correct_only else ''}")

    plot_bar(list(step_raw.keys()), list(step_raw.values()), xlabel="Step ID", ylabel="Average Raw",
             title=f"Average Raw{' Correct' if args.correct_only else ''}")
    plot_bar(list(step_prob.keys()), list(step_prob.values()), xlabel="Step ID", ylabel="Average Prob",
             title=f"Average Prob{' Correct' if args.correct_only else ''}")


if __name__ == "__main__":
    main()
