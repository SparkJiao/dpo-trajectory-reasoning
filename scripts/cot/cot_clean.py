import argparse
import json

from nltk import sent_tokenize
from tqdm import tqdm


# def response_clean(response: str):
#     lines = response.split("\n")
#     lines = [line.strip() for line in lines]
#     lines = [line for line in lines if line]
#     steps = []
#     for line_id, line in enumerate(lines):
#         sentences = sent_tokenize(line)
#
#         line_steps = []
#         for i, t in enumerate(sentences):
#             if t.strip() == "":
#                 continue
#             if len(line_steps) == 0:
#                 line_steps.append(t)
#             else:
#                 line_steps.append(" " + t)
#
#         if len(line_steps) > 0 and line_id != len(lines) - 1:
#             line_steps[-1] = line_steps[-1] + "\n"
#             steps.extend(line_steps)
#     return steps


def response_clean(response: str):
    lines = response.split("\n")
    steps = []
    for i, line in enumerate(lines):
        if line.strip():
            sentences = sent_tokenize(line)
            tmp = []
            for j, sent in enumerate(sentences):
                if j == 0:
                    tmp.append(sent)
                else:
                    tmp.append(" " + sent)
            steps.append(tmp)
        else:
            steps.append(line)

    results = []
    for i, x in enumerate(steps):
        if isinstance(x, str):
            x = [x]
        if i != 0:
            x[0] = "\n" + x[0]
        results.extend(x)

    steps = []
    for step in results:
        if len(steps) == 0 or steps[-1].strip():
            steps.append(step)
        elif steps[-1].strip() == "":
            steps[-1] += step
        else:
            raise NotImplementedError(f"steps[-1]: {steps[-1]}")

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    parser.add_argument("--orig_data_file", type=str, required=True)
    parser.add_argument("--response_field", type=str, default="response")
    parser.add_argument("--id_field", type=str, default="id")
    args = parser.parse_args()

    responses = json.load(open(args.response_file))
    orig_data = json.load(open(args.orig_data_file))

    id2response = {item["id"]: item for item in responses}
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

        responses = responses_tmp
        preds = preds_tmp

        tmp_resp = []
        tmp_pred = []
        for resp, pred in zip(responses, preds):
            res = response_clean(resp)
            if res:
                tmp_resp.append(res)
                tmp_pred.append(pred)
        if tmp_resp:
            item[args.response_field] = tmp_resp
            item["pred"] = tmp_pred
            outputs.append(item)

    print(f"Filtered {len(outputs)} items")

    save_file = args.response_file.replace(".json", "_clean.json")
    json.dump(outputs, open(save_file, "w"), indent=2)


if __name__ == '__main__':
    main()
