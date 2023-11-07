import argparse
import json
import re
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
args = parser.parse_args()

data = json.load(open(args.input_file))

outputs = []

for item in data:
    chosen = []
    reject = []
    for response in item["response"]:
        if "[Context]" in response:
            response = response.split("[Context]")[0]

        preds = re.findall(r"A|B|C|D", response)
        if len(preds) == 0:
            pred = ""
        else:
            pred = preds[-1]

        if pred and ord(pred) - ord("A") == item["label"]:
            chosen.append(response)
        else:
            reject.append(response)

    if len(chosen) > 0 and len(reject) > 0:
        outputs.append({
            "input": item["text"],
            "chosen": chosen,
            "reject": reject,
            "id": item["id"],
        })

print(len(outputs))

a_cnt = Counter()
b_cnt = Counter()
for x in outputs:
    a_cnt[len(x["chosen"])] += 1
    b_cnt[len(x["reject"])] += 1
print(a_cnt)
print(b_cnt)

json.dump(outputs, open(args.output_file, "w"), indent=2, ensure_ascii=False)
