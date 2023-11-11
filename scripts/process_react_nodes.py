import random
import re
import argparse
import json


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


def process_item(item):
    response = item["response"]
    if isinstance(response, str):
        response = [response]
    all_nodes = []
    # idx_pattern = re.compile(r'\d+\w*')
    for res in response:
        lines = process_response(res)
        nodes = []
        for line in lines:
            if line.startswith("Thought "):
                pattern = re.compile(r'Thought \S+:')
                node_type = "Thought"
            elif line.startswith("Action "):
                pattern = re.compile(r'Action \S+:')
                node_type = "Action"
            elif line.startswith("Observation "):
                pattern = re.compile(r'Observation \S+:')
                node_type = "Observation"
            else:
                raise ValueError(line)

            m = pattern.match(line)
            if m is None:
                continue
            # assert m is not None, (line, "\n".join(lines))
            assert m.span(0)[0] == 0
            # idx = idx_pattern.findall(m.group(0))
            # assert len(idx), (line, lines)
            idx = m.group(0)[len(node_type) + 1: -1]
            nodes.append({
                "id": idx,
                "type": node_type,
                "content": line[m.span(0)[1]:].strip()
            })
        assert "Finish[The answer is" in nodes[-1]["content"], (res, lines, nodes[-1]["content"])
        all_nodes.append(nodes)
    return all_nodes


def process_response(response: str):
    lines = response.split("\n")
    lines = list(filter(lambda x: x.startswith("Thought ") or x.startswith("Action ") or x.startswith("Observation "), lines))
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    # Clean
    cleaned_data = []
    cnt_a = 0
    cnt_b = 0
    for item in data:
        response = item["response"]
        preds = item["pred"]
        if isinstance(response, str):
            response = [response]
            preds = [preds]
        new_responses = []
        new_preds = []
        for res, pred in zip(response, preds):
            cnt_a += 1
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
        cnt_b += len(new_responses)

    print(f"Clean: {cnt_a} -> {cnt_b}")

    # Process
    for item in cleaned_data:
        item["nodes"] = process_item(item)

    json.dump(cleaned_data, open(args.output_file, "w"), indent=2)
    samples = random.sample(cleaned_data, 1000)
    json.dump(samples, open(args.output_file[:-len(".json")] + ".sample.json", "w"), indent=2)


if __name__ == '__main__':
    main()
