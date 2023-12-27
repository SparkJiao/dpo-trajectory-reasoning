import argparse
import json
import re
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

data = json.load(open(args.input_file, "r"))

cnt = 0
tmp = 0
warning = 0
hacked = 0
hacked_responses = []
for item in data:
    response = item["response"]
    if "Finish[" in response:
        tmp += 1
        groups = response.split("Finish[")
        if len(groups) > 2:
            print(response)
            print("Label: ", item["label"])
            print("=========================")
        # response = response.split("Finish[")[1]
        response = groups[1]

    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        pred = ""
    elif len(preds) > 1:
        pred = ""
        hacked += 1
        if "The answer is" in response:
            hacked_responses.append(response)
    else:
        pred = preds[0]

    if pred and ord(pred) - ord("A") == item["label"]:
        cnt += 1

print(cnt / len(data))
print("Finished: ", tmp)
print("Hacked answer: ", hacked)
print(len(data))
print(json.dumps(hacked_responses[:50], indent=2))
