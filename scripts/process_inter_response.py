import argparse
import collections
import json
import os
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--diff", type=int, default=2)
    parser.add_argument("--inter_state_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        files = [args.input_file]
    else:
        files = glob(args.input_file)
    print(files)

    state_id2values = collections.defaultdict(dict)
    for file in files:
        data = json.load(open(file, "r"))

        for item in data:
            idx = item["id"]
            idx, state_id = idx.split("_")
            v = 0
            flag = True
            for res, p in zip(item["response"], item["pred"]):
                if res == "":
                    flag = False
                    break
                if p == "":
                    continue
                if ord(p.strip()) - ord("A") == item["label"]:
                    v += 1
            if not flag:
                continue
            state_id2values[int(idx)][int(state_id)] = (v, item["text"].split("Thought 1:")[1].strip())

    max_res_num = max([len(x) for x in state_id2values.values()])
    state_id2values = {k: v for k, v in state_id2values.items() if len(v) == max_res_num}
    print(len(state_id2values))

    if os.path.exists(args.inter_state_file):
        inter_state_files = [args.inter_state_file]
    else:
        inter_state_files = glob(args.inter_state_file)
    inter_states = []
    for state_file in inter_state_files:
        inter_states.extend(json.load(open(state_file, "r")))

    outputs = []
    jumped = 0
    for item in inter_states:
        idx = item["id"]
        if idx not in state_id2values:
            jumped += 1
            continue

        item_states = item["inter_states"]
        for x_i, x in enumerate(item_states):
            val_x, res_x = state_id2values[idx][x_i]
            assert res_x in item_states[x_i], (res_x, item_states[x_i])
            for y_i, y in enumerate(item_states):
                if x_i == y_i:
                    continue
                val_y, res_y = state_id2values[idx][y_i]
                assert res_y in item_states[y_i], (res_y, item_states[y_i])
                if val_x - val_y >= args.diff:
                    outputs.append({
                        "id": idx,
                        "chosen": x,
                        "reject": y,
                        "is_full": False,
                    })

    print(f"Jumped: {jumped}")
    print(len(outputs))
    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
