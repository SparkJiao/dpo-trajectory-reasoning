import json
import argparse
import re
from typing import List, Dict, Union, Tuple
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
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            return 0


def parse_step_samples_w_value(response: str, nodes: List[Dict[str, Union[str, float]]], included_types: Tuple[str] = ("Thought", "Observation")):
    raw_steps = response.split("\n")

    accumulated_step_samples = []
    node_pointer = 0
    for step_id, step in enumerate(raw_steps):
        if nodes[node_pointer]["content"] in step:
            if nodes[node_pointer]["type"] in included_types:
                sub_sample_content = "\n".join(raw_steps[:step_id])
                accumulated_step_samples.append({
                    "content": sub_sample_content,
                    "value": nodes[node_pointer]["value"],
                    "length": step_id + 1,
                })
            node_pointer += 1
            if node_pointer == len(nodes):
                break

    if node_pointer == 0:
        raise RuntimeError(f"Warning: {response} does not contain any node in {included_types}. Actions: {nodes}")

    return accumulated_step_samples


def extract_pairwise_step_samples(data_id: Union[int, str],
                                  resp_a_samples: List[Dict[str, Union[str, float]]],
                                  resp_b_samples: List[Dict[str, Union[str, float]]],
                                  step_lens_diff: int,
                                  max_inter_samples: int,
                                  value_diff: float):
    outputs = []
    a_pointer = 0
    b_pointer = 0
    while a_pointer < len(resp_a_samples) and b_pointer < len(resp_b_samples):
        if abs(resp_a_samples[a_pointer]["length"] - resp_b_samples[b_pointer]["length"]) > step_lens_diff:
            if resp_a_samples[a_pointer]["length"] < resp_b_samples[b_pointer]["length"]:
                a_pointer += 1
            else:
                b_pointer += 1
            continue

        if abs(resp_a_samples[a_pointer]["value"] - resp_b_samples[b_pointer]["value"]) < value_diff:
            if resp_a_samples[a_pointer]["length"] < resp_b_samples[b_pointer]["length"]:
                a_pointer += 1
            elif resp_a_samples[a_pointer]["length"] > resp_b_samples[b_pointer]["length"]:
                b_pointer += 1
            elif len(resp_a_samples) < len(resp_b_samples):
                b_pointer += 1
            else:
                a_pointer += 1
            continue

        if len(outputs) >= max_inter_samples:
            break

        if resp_a_samples[a_pointer]["value"] > resp_b_samples[b_pointer]["value"]:
            chosen = resp_a_samples[a_pointer]["content"]
            rejected = resp_b_samples[b_pointer]["content"]
        else:
            chosen = resp_b_samples[b_pointer]["content"]
            rejected = resp_a_samples[a_pointer]["content"]

        outputs.append({
            "chosen": chosen,
            "reject": rejected,
            "id": data_id
        })
        a_pointer += 1
        b_pointer += 1

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--step_lens_diff", type=int, default=2)
    parser.add_argument("--max_inter_samples", type=int, default=4)
    parser.add_argument("--value_diff", type=int, default=0.1)
    parser.add_argument("--included_types", type=str, nargs="+", default=("Thought", "Observation"))
    parser.add_argument("--save_full_data", action="store_true", default=False)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    outputs = []
    full_samples = []
    avg_step_sample_num = 0
    for item in data:
        final_chosen = []
        final_rejected = []
        for response_id, response in enumerate(item["response"]):
            v = parse_leaf_node_value(response, item["label"])
            if v == 0:
                final_rejected.append(response_id)
            else:
                final_chosen.append(response_id)

        if len(final_chosen) == 0 or len(final_rejected) == 0:
            continue

        resp_acc_step_samples = []
        for response_id, response in enumerate(item["response"]):
            resp_acc_step_samples.append(parse_step_samples_w_value(response, item["nodes"][response_id], args.included_types))

        step_samples = []
        for resp_i in final_chosen:
            for resp_j in final_rejected:
                res = extract_pairwise_step_samples(item["id"],
                                                    resp_acc_step_samples[resp_i],
                                                    resp_acc_step_samples[resp_j],
                                                    args.step_lens_diff,
                                                    args.max_inter_samples,
                                                    args.value_diff, )
                step_samples.extend(res)

                full_res = {
                    "chosen": item["response"][resp_i],
                    "reject": item["response"][resp_j],
                    "id": item["id"],
                }
                outputs.append(full_res)
                full_samples.append(full_res)

        avg_step_sample_num += len(step_samples)
        outputs.extend(step_samples)

    print(f"Average step sample number: {avg_step_sample_num / len(data)}")
    print(f"Average sample numer: {len(outputs) / len(data)}")
    print(len(outputs))

    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)
    if args.save_full_data:
        print(f"Save full data to {args.output_file.replace('.json', '.full_only.json')}")
        print(f"Full data number: {len(full_samples)}")
        json.dump(full_samples, open(args.output_file.replace(".json", ".full_only.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
