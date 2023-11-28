import argparse
import collections
import json
import os
from glob import glob
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--diff", type=float, default=2.4)
    parser.add_argument("--inter_state_file", type=str)
    parser.add_argument("--decay", type=float, default=0.9)
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
    for item in tqdm(inter_states, total=len(inter_states)):
        idx = item["id"]
        if idx not in state_id2values:
            jumped += 1
            continue

        item_states = item["inter_states"]
        resp_id2states = collections.defaultdict(list)
        for x_id, x in enumerate(item_states):
            if x_id not in state_id2values[idx]:
                continue
            resp_id2states[x["resp_id"]].append((x, x_id))

        for x_i, x in enumerate(item_states):
            if x_i not in state_id2values[idx]:
                continue
            step_id_x = x["step_id"]
            val_x, res_x = state_id2values[idx][x_i]
            assert res_x in x["state"], (res_x, item_states[x_i])
            for y_i, y in enumerate(item_states):
                if x_i == y_i:
                    continue
                if y_i not in state_id2values[idx]:
                    continue
                step_id_y = y["step_id"]
                val_y, res_y = state_id2values[idx][y_i]
                assert res_y in y["state"], (res_y, item_states[y_i])
                if step_id_x < step_id_y:
                    val_x = val_x * args.decay ** (step_id_y - step_id_x)
                elif step_id_x > step_id_y:
                    val_y = val_y * args.decay ** (step_id_x - step_id_y)
                if val_x - val_y >= args.diff:
                    outputs.append({
                        "id": idx,
                        "chosen": x["state"],
                        "reject": y["state"],
                        "is_full": False,
                        "val_diff": val_x - val_y,
                    })

    print(f"Jumped: {jumped}")
    print(len(outputs))
    json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
