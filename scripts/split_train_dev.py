import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--dev_num", type=int)
args = parser.parse_args()

data = json.load(open(args.input_file))
data_ids = list(range(len(data)))
dev_ids = random.sample(data_ids, args.dev_num)
dev_ids = set(dev_ids)

dev_data = []
train_data = []
for i, item in enumerate(data):
    if i in dev_ids:
        dev_data.append(item)
    else:
        train_data.append(item)

json.dump(dev_data, open(args.input_file.replace(".json", ".sub_dev.json"), "w"), indent=2, ensure_ascii=False)
json.dump(train_data, open(args.input_file.replace(".json", ".sub_train.json"), "w"), indent=2, ensure_ascii=False)
