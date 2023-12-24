import argparse
import json
import os
import re
from glob import glob
from typing import Dict


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
            duplicates.add(item["response"])
            cnt += 1
        if isinstance(item["reward"], list):
            assert len(item["reward"]) == 1
            response2reward[item["response"]] = item["reward"][0]
        elif isinstance(item["reward"], float):
            response2reward[item["response"]] = item["reward"]
        else:
            raise ValueError(f"Unsupported type of reward: {item['reward'].__class__}")

    print("collected rewards", len(response2reward))
    print("duplicate responses", cnt)

    for item in data:
        cleaned_responses = []
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
        item["response_reward"] = cleaned_responses

    json.dump(data, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
