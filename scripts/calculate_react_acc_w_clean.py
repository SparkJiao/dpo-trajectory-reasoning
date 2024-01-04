import argparse
import json
import re
from collections import Counter
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_processors.openai_api_callback import ReActSeparatorClean

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--save_numpy", default=False, action="store_true")
args = parser.parse_args()

data = json.load(open(args.input_file, "r"))

# cnt = 0
# tmp = 0
# warning = 0
# hacked = 0
# hacked_responses = []
# no_ending_results = []
# for item in data:
#     response = item["response"]
#     if "Finish[" in response:
#         tmp += 1
#         groups = response.split("Finish[")
#         if len(groups) > 2:
#             print(response)
#             print("Label: ", item["label"])
#             print("=========================")
#         # response = response.split("Finish[")[1]
#         response = groups[1]
#     else:
#         warning += 1
#         no_ending_results.append(response)
#
#     preds = re.findall(r"A|B|C|D", response)
#     if len(preds) == 0:
#         pred = ""
#     elif len(preds) > 1:
#         pred = ""
#         if "The answer is" in response:
#             hacked += 1
#             hacked_responses.append(response)
#     else:
#         pred = preds[0]
#
#     if pred and ord(pred) - ord("A") == item["label"]:
#         cnt += 1
#
# print(cnt / len(data))
# print("Finished: ", tmp)
# print("Hacked answer: ", hacked)
# print(len(data))
# print(warning)
# print(json.dumps(hacked_responses[:50], indent=2))
# print(json.dumps(no_ending_results[:50], indent=2))

cleaner = ReActSeparatorClean()

errors = []
cnt = 0
outputs = []
for item in data:
    response = item["response"]

    pred = cleaner(response)
    if pred == "":
        errors.append(response)
        outputs.append((item["id"], 0))
        continue

    if ord(pred) - ord("A") == item["label"]:
        cnt += 1
    outputs.append((item["id"], ord(pred) - ord("A")))

assert len(outputs) == len(data)
if args.save_numpy:
    import numpy as np

    # Remove duplicated ids to satisfy the submission requirements of ReClor.
    outputs = sorted(outputs, key=lambda x: x[0])
    id_set = set()
    new_outputs = []
    for item in outputs:
        if item[0] not in id_set:
            new_outputs.append(item[1])
            id_set.add(item[0])
    outputs = new_outputs
    np.save(args.input_file.replace(".json", ".clean.npy"), np.array(outputs))

if args.debug:
    print(json.dumps(errors[:50], indent=2))
print(cnt / len(data))
print(len(errors))
