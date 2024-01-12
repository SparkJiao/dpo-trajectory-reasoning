import argparse
import json
import os.path
import re
from glob import glob


# Copied from scripts.process_react_nodes.py
# Update 2023/12/27: Remove the falsehood solutions with predicting multiple answers, e.g., Finish[The answer is A and B].
#   This would lead to the model prefer to hack problem.
# Update 2024/01/08: Merge multiple lines and remove blank lines.
# Update 2024/01/08: Remove duplicate responses.
def clean_react_response(response: str):
    if "Context:\n" in response:
        response = response.split("Context:\n")[0]

    lines = response.split("\n")
    finish_lines = [lines for line in lines if "Finish[The answer is" in line and line.startswith("Action ")]
    if len(finish_lines) != 1:
        return None

    new_lines = []
    for line in lines:
        if line.strip() == "":
            continue
        if not (line.startswith("Thought ") or line.startswith("Action ") or line.startswith("Observation ")):
            if len(new_lines) > 0:
                new_lines[-1] += " " + line
                continue
        else:
            if line.startswith("Thought "):
                content = line.split("Thought ")[1]
            elif line.startswith("Action "):
                content = line.split("Action ")[1]
            elif line.startswith("Observation "):
                content = line.split("Observation ")[1]
            else:
                raise NotImplementedError
            if len(content) < 5:
                continue
        new_lines.append(line)
        if "Finish[The answer is" in line and line.startswith("Action "):
            break
    # assert "Finish[The answer is" in new_lines[-1] and new_lines[-1].startswith("Action "), (new_lines, lines)
    if "Finish[The answer is" not in new_lines[-1] or not new_lines[-1].startswith("Action "):
        return None
    response = "\n".join(new_lines)
    response = response.strip()
    return response


def parse_leaf_node_value(response: str, label: int, logs: dict):
    groups = response.split("Finish")
    if len(groups) < 2:
        # print(f"Warning: Not a valid response: {response}")
        logs["invalid"] += 1
        return 0
    response = groups[1]
    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        logs["missing"] += 1
        return 0
    elif len(preds) > 1:
        logs["multiple"] += 1
        # print(f"Warning: Multiple answers: {response}")  # Fixed: Here is fixed.
        return 0
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            logs["wrong"] += 1
            return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--min_step", type=int, default=0)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    print(files)
    data = []
    for file in files:
        data.extend(json.load(open(file)))
    print(len(data))

    cleaned_data = []
    avg_resp_num = 0
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
            if len(res.split("\n")) < args.min_step:
                continue
            if res not in new_responses:
                new_responses.append(res)
                new_preds.append(pred)
        if len(new_responses) == 0:
            continue
        item["response"] = new_responses
        item["pred"] = new_preds
        cleaned_data.append(item)
        avg_resp_num += len(new_responses)

    print(f"Clean: {len(data)} -> {len(cleaned_data)}")
    print(f"Average response number: {avg_resp_num / len(cleaned_data)}")

    outputs = []
    all_chosen_data = []
    logs = {
        "invalid": 0,
        "missing": 0,
        "multiple": 0,
        "wrong": 0,
    }
    for item in cleaned_data:
        chosen = []
        reject = []
        for response in item["response"]:
            v = parse_leaf_node_value(response, item["label"], logs=logs)
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
    print(logs)

    save_dir = os.path.dirname(args.output_file)
    cleaned_data_save_path = os.path.join(save_dir, args.input_file.split("/")[-1].replace(".json", ".cleaned.json"))
    json.dump(cleaned_data, open(cleaned_data_save_path, "w"), indent=2, ensure_ascii=False)
    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)
    json.dump(all_chosen_data, open(args.output_file.replace(".json", ".chosen.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
