import sys
import json
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.math import math_answer_cleaner

parser = argparse.ArgumentParser()
parser.add_argument("--response_file", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

data = json.load(open(args.response_file))
cleaner = math_answer_cleaner(separator="The answer is")

for item in data:
    new_preds = []
    for resp in item["response"]:
        new_preds.append(cleaner(resp))

    item["pred"] = new_preds

json.dump(data, open(args.output_file, "w"), indent=2)
