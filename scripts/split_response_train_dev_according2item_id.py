import argparse
import collections
import json
import os.path
import random
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    item_id2responses = collections.defaultdict(list)
    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        files = glob(args.input_file)
        print(files)
        data = []
        for file in files:
            data.extend(json.load(open(file)))

    for item in data:
        item_id, state_id = item['id'].split("_")
        item_id2responses[item_id].append(item)

    print("data size: {}".format(len(item_id2responses)))
    print(f"Response size: {len(data)}")

    data_ids = list(item_id2responses.keys())
    # read `dev_num` from command line
    dev_num = int(input("dev_num: "))
    dev_ids = random.sample(data_ids, dev_num)
    dev_ids = set(dev_ids)

    dev_data = []
    train_data = []
    for item_id, responses in item_id2responses.items():
        if item_id in dev_ids:
            dev_data.extend(responses)
        else:
            train_data.extend(responses)

    print("dev size: {}".format(len(dev_data)))
    print("train size: {}".format(len(train_data)))

    json.dump(dev_data, open(args.output_file.replace(".json", f".sub_dev.{len(dev_data)}.json"), "w"), indent=2, ensure_ascii=False)
    json.dump(train_data, open(args.output_file.replace(".json", f".sub_train.{len(train_data)}.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
