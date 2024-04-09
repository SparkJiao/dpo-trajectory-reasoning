import argparse
import json
import os.path
from glob import glob

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    parser.add_argument("--orig_data_file", type=str, required=True)
    parser.add_argument("--response_field", type=str, default="response")
    parser.add_argument("--id_field", type=str, default="id")
    args = parser.parse_args()

    if os.path.exists(args.response_file):
        files = [args.response_file]
    else:
        files = glob(args.response_file)
        assert len(files) > 0, f"No files found in {args.response_file}"
    responses = []
    for file in files:
        responses.extend(json.load(open(file)))
    orig_data = json.load(open(args.orig_data_file))

    id2response = {item["id"]: item for item in responses}
    print(len(id2response))
    outputs = []
    for i, item in tqdm(enumerate(orig_data), desc="Cleaning responses", total=len(orig_data)):
        if args.id_field not in item:
            item[args.id_field] = i
        if item[args.id_field] not in id2response:
            continue
        responses = id2response[item[args.id_field]]["response"]
        preds = id2response[item[args.id_field]]["pred"]

        if isinstance(responses, str):
            responses = [responses]
            preds = [preds]

        set_tmp = set()
        responses_tmp = []
        preds_tmp = []
        for resp, pred in zip(responses, preds):
            if not resp.strip():
                continue
            if resp not in set_tmp:
                set_tmp.add(resp)
                responses_tmp.append(resp)
                preds_tmp.append(pred)

        if len(responses_tmp) == 0:
            continue

        item[args.response_field] = responses_tmp
        item["pred"] = preds_tmp
        item["label"] = id2response[item[args.id_field]]["label"]
        outputs.append(item)

    print(f"Filtered {len(outputs)} items")

    tmp = args.response_file.split(".")
    tmp = [x for x in tmp if "?" not in x and "*" not in x]
    tmp = ".".join(tmp)

    save_file = tmp.replace(".json", "_clean.json")
    json.dump(outputs, open(save_file, "w"), indent=2)


if __name__ == '__main__':
    main()
