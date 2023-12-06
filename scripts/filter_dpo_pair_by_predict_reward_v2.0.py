import json
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file1", type=str)
    parser.add_argument("--input_file2", type=str, default=None)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--debug_file", type=str, default=None)
    args = parser.parse_args()

    files = [args.input_file1]
    if args.input_file2 is not None:
        files.append(args.input_file2)
    data = []
    for file in files:
        data.extend(json.load(open(file, "r")))
    print(len(data))

    rewards = json.load(open(args.reward_file, "r"))
    response2reward = {}
    duplicates = set()
    cnt = 0
    for item in rewards:
        if item["response"] in response2reward:
            # print("duplicate response")
            duplicates.add(item["response"])
            cnt += 1
        assert len(item["reward"]) == 1
        response2reward[item["response"]] = item["reward"][0]

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)

    uncovered = 0
    filtered = []
    for item in data:
        chosen = item["chosen"]
        if "</s>" in chosen:
            chosen = chosen.replace("</s>", "")
        reject = item["reject"]
        if "</s>" in reject:
            reject = reject.replace("</s>", "")
        if chosen not in response2reward or reject not in response2reward:
            uncovered += 1
            continue
        chosen_reward = response2reward[chosen]
        reject_reward = response2reward[reject]
        item["chosen_reward"] = chosen_reward
        item["reject_reward"] = reject_reward
        if chosen_reward - reject_reward >= args.margin:
            filtered.append(item)

    print("Uncovered responses", uncovered)
    print("Collected amount of samples with rewards", len(filtered))
    json.dump(filtered, open(args.output_file, "w"), indent=4)

    if args.debug_file is not None:
        json.dump(data, open(args.debug_file, "w"))


if __name__ == '__main__':
    main()
