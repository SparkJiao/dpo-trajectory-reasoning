import argparse
import json
import os
from glob import glob
from typing import Dict


def best_of_n_filter(item, best_of: int, response2reward: Dict[str, float]):
    best_responses = []
    for resp_id, resp in enumerate(item["response"]):
        if resp not in response2reward:
            continue
        best_responses.append((resp_id, response2reward[resp]))

    best_responses = sorted(best_responses, key=lambda x: x[1], reverse=True)
    best_responses = best_responses[:best_of]
    best_responses = sorted(best_responses, key=lambda x: x[0])
    best_responses = [item["response"][resp_id] for resp_id, _ in best_responses]
    return best_responses


def best_of_n_filter_inter_states(item, best_of: int, response2reward: Dict[str, float]):
    best_responses = []
    for state_id, state in enumerate(item["inter_states"]):
        resp = state["state"]
        if resp not in response2reward:
            continue
        best_responses.append((state_id, response2reward[resp]))

    best_responses = sorted(best_responses, key=lambda x: x[1], reverse=True)
    best_responses = best_responses[:best_of]
    best_responses = sorted(best_responses, key=lambda x: x[0])
    best_responses = [item["inter_states"][state_id]["state"] for state_id, _ in best_responses]
    return best_responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--reward_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--best_of", type=int, default=1)
    parser.add_argument("--inter_best_of", type=int, default=1)
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
        assert len(item["reward"]) == 1
        response2reward[item["response"]] = item["reward"][0]

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)

    filtered = []
    for item in data:
        responses = best_of_n_filter(item, args.best_of, response2reward)
        for resp in responses:
            filtered.append({
                "chosen": resp,
                "id": item["id"],
                "is_full": True,
            })

        responses = best_of_n_filter_inter_states(item, args.inter_best_of, response2reward)
        for resp in responses:
            filtered.append({
                "chosen": resp,
                "id": item["id"],
                "is_full": False,
            })

    print("Collected amount of samples with rewards", len(filtered))
    json.dump(filtered, open(args.output_file, "w"), indent=4)


if __name__ == '__main__':
    main()
