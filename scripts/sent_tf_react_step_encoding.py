import argparse
from FlagEmbedding import FlagModel
import json
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = json.load(open(args.input_file, "r"))

    node_ids = []
    node_types = []
    node_rationales = []
    node2offset = {}
    node2idx = {}
    for j, item in tqdm(enumerate(data)):
        node2offset[j] = [len(node_rationales), -1]
        node2idx[j] = {}
        for i in range(len(item["response"])):
            chain_nodes = item["nodes"][i]

            chain_node_ids = [node["id"] for node in chain_nodes]
            chain_node_types = [node["type"] for node in chain_nodes]
            chain_node_rationales = [node["content"] for node in chain_nodes]

            node2idx[j][i] = len(node_ids)

            node_ids.extend(chain_node_ids)
            node_types.extend(chain_node_types)
            node_rationales.extend(chain_node_rationales)

            assert "Finish[The answer is" in item["nodes"][i][-1]["content"]

        node2offset[j][1] = len(node_rationales)

    model = FlagModel(args.model_path,
                      # query_instruction_for_retrieval="",
                      use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
    embeddings = model.encode(node_rationales)

    np.save(args.output_file, embeddings)


if __name__ == '__main__':
    main()
