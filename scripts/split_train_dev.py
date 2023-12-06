import json
import argparse
import random
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--input_file2", type=str, default=None)
args = parser.parse_args()

data = json.load(open(args.input_file))
print("data size: {}".format(len(data)))
if args.input_file2 is not None:
    data2 = json.load(open(args.input_file2))
    print("data2 size: {}".format(len(data2)))
data_ids = list(range(len(data)))
# read `dev_num` from command line
dev_num = int(input("dev_num: "))
dev_ids = random.sample(data_ids, dev_num)
dev_ids = set(dev_ids)

dev_data = []
train_data = []
for i, item in enumerate(data):
    if i in dev_ids:
        dev_data.append(item)
    else:
        train_data.append(item)

print("dev size: {}".format(len(dev_data)))
print("train size: {}".format(len(train_data)))

if args.input_file2 is not None:
    output_file_name = str(input("output file name: "))
    output_file = os.path.join(os.path.dirname(args.input_file), output_file_name)
else:
    output_file = args.input_file
json.dump(dev_data, open(output_file.replace(".json", ".sub_dev.json"), "w"), indent=2, ensure_ascii=False)
json.dump(train_data, open(output_file.replace(".json", ".sub_train.json"), "w"), indent=2, ensure_ascii=False)
