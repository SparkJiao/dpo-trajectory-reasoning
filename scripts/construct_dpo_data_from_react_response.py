import argparse
import json
import os.path
import re
from glob import glob


# Copied from scripts.process_react_nodes.py
def clean_react_response(response: str):
    if "Context:\n" in response:
        response = response.split("Context:\n")[0]

    lines = response.split("\n")
    finish_lines = [lines for line in lines if "Finish[The answer is" in line and line.startswith("Action ")]
    if len(finish_lines) != 1:
        return None

    new_lines = []
    for line in lines:
        new_lines.append(line)
        if "Finish[The answer is" in line and line.startswith("Action "):
            break
    assert "Finish[The answer is" in new_lines[-1] and new_lines[-1].startswith("Action "), new_lines
    response = "\n".join(new_lines)
    response = response.strip()
    return response


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    print(files)
    data = []
    for file in files:
        data.extend(json.load(open(file)))
    # data = json.load(open(args.input_file))
    print(len(data))

    cleaned_data = []
    for item in data:
        response = item["response"]
        preds = item["pred"]
        if isinstance(response, str):
            response = [response]
            preds = [preds]
        new_responses = []
        new_preds = []
        for res, pred in zip(response, preds):
            res = clean_react_response(res)
            if res is None:
                continue
            new_responses.append(res)
            new_preds.append(pred)
        if len(new_responses) == 0:
            continue
        item["response"] = new_responses
        item["pred"] = new_preds
        cleaned_data.append(item)

    print(f"Clean: {len(data)} -> {len(cleaned_data)}")

    outputs = []
    all_chosen_data = []
    for item in cleaned_data:
        chosen = []
        reject = []
        for response in item["response"]:
            v = parse_leaf_node_value(response, item["label"])
            if v == 0:
                reject.append(response)
            else:
                chosen.append(response)

        if len(chosen) > 0 and len(reject) > 0:
            for resp_i in chosen:
                for resp_j in reject:
                    outputs.append({
                        "chosen": resp_i,
                        "reject": resp_j,
                        "id": item["id"],
                    })

        if len(chosen) > 0:
            for resp in chosen:
                all_chosen_data.append({
                    "chosen": resp,
                    "id": item["id"],
                })

    print(len(all_chosen_data))
    print(len(outputs))

    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)
    json.dump(all_chosen_data, open(args.output_file.replace(".json", ".chosen.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
