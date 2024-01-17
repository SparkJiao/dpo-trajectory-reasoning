import json
import argparse
import random
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    args = parser.parse_args()

    data = json.load(open(args.input_file, "r"))
    print(len(data))

    id2samples = defaultdict(list)
    for item in data:
        id2samples[item["id"]].append(item)

    print(len(id2samples))

    sampled_data_ids = random.sample(list(id2samples.keys()), int(len(id2samples) * args.ratio))
    sampled_data = []
    for sample_id in sampled_data_ids:
        sampled_data.extend(id2samples[sample_id])
    print(len(sampled_data))

    json.dump(sampled_data, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
