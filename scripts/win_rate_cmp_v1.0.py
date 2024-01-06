import argparse
import os
import json
from collections import Counter
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    args = parser.parse_args()

    if args.input_file.endswith(".json"):
        data = json.load(open(args.input_file))
    else:
        data = [json.loads(line) for line in open(args.input_file).readlines()]

    reasonable = Counter()
    concise = Counter()
    consistent = Counter()
    overall = Counter()

    missed_reasonable = 0
    missed_concise = 0
    missed_consistent = 0
    missed_overall = 0
    for item in data:
        response = item["response"]

        if "Reasonable:" in response:
            tmp_a = response.split("Reasonable:")[1]
            ans = re.findall(r"A|B|Tie", tmp_a)[0]
            reasonable[ans] += 1
        else:
            missed_reasonable += 1

        if "Concise:" in response:
            tmp_a = response.split("Concise:")[1]
            ans = re.findall(r"A|B|Tie", tmp_a)[0]
            concise[ans] += 1
        else:
            missed_concise += 1

        if "Logically consistent:" in response:
            tmp_a = response.split("Logically consistent:")[1]
            ans = re.findall(r"A|B|Tie", tmp_a)[0]
            consistent[ans] += 1
        else:
            missed_consistent += 1

        if "Overall:" in response:
            tmp_a = response.split("Overall:")[1]
            ans = re.findall(r"A|B|Tie", tmp_a)[0]
            overall[ans] += 1
        else:
            missed_overall += 1

    print(f"Reasonable: {reasonable}")
    print(f"Concise: {concise}")
    print(f"Consistent: {consistent}")
    print(f"Overall: {overall}")

    print(f"Missed Reasonable: {missed_reasonable}")
    print(f"Missed Concise: {missed_concise}")
    print(f"Missed Consistent: {missed_consistent}")
    print(f"Missed Overall: {missed_overall}")


if __name__ == "__main__":
    main()
