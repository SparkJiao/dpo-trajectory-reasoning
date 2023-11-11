import argparse
import collections
import json
from tqdm import tqdm
import numpy as np
from typing import List, Any, Dict, Tuple
import re


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x == y:
            return
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parent[y] = x
        self.size[x] += self.size[y]


def node_clustering(node_embeddings: np.array, node_ids: List[str], id2node, threshold: float = 0.9, included_types: Tuple[str] = ("Thought", "Observation")):
    # TODO: If add constraint here to ensure that the leaf nodes are not connected to each other.
    #   Note that, considering that not all leaf nodes are action nodes, considering the unstable generation process.
    # FIXME: 1. Remove the leaf nodes in clustering.
    node_type2idx = {}
    for i, node_unique_id in enumerate(node_ids):
        node_type = node_unique_id.split("$")[1]
        if node_type not in node_type2idx:
            node_type2idx[node_type] = []
        node_type2idx[node_type].append(i)

    cluster = UnionFind(len(node_ids))
    for node_type in included_types:
        node_type_idx = node_type2idx[node_type]
        node_type_embeddings = node_embeddings[node_type_idx]

        node_type_similarity = node_type_embeddings @ node_type_embeddings.T

        for i in range(len(node_type_idx)):
            if id2node[node_ids[node_type_idx[i]]]["is_leaf"]:
                continue
            response_id_i = int(node_ids[node_type_idx[i]].split("$")[0])
            for j in range(i + 1, len(node_type_idx)):
                if id2node[node_ids[node_type_idx[j]]]["is_leaf"]:
                    continue
                response_id_j = int(node_ids[node_type_idx[j]].split("$")[0])
                if response_id_i == response_id_j:
                    continue
                if node_type_similarity[i, j] > threshold:
                    cluster.union(node_type_idx[i], node_type_idx[j])

    node2cluster = {}
    for i, node_unique_id in enumerate(node_ids):
        node2cluster[node_unique_id] = cluster.find(i)

    return node2cluster


def get_node_unique_id(response_id, node):
    return "$".join([str(response_id), node["type"], node["id"]])


def parse_leaf_node_value(response, label):
    groups = response.split("Finish")
    if len(groups) < 2:
        print(f"Warning: Not a valid response: {response}")
        return 0
    response = groups[1]
    preds = re.findall(r"A|B|C|D", response)
    if len(preds) == 0:
        return 0
    elif len(preds) > 1:
        return 0
    else:
        if ord(preds[0]) - ord("A") == label:
            return 1
        else:
            return 0


def dfs(tree, cluster_id, cluster2value, cluster2nodes, label):
    if cluster_id not in tree:
        if cluster_id not in cluster2value:
            value = 0
            for leaf_node in cluster2nodes[cluster_id]:
                assert leaf_node["is_leaf"], f"Warning: {leaf_node} is not a leaf node."
                value += parse_leaf_node_value(leaf_node["content"], label)
            value /= len(cluster2nodes[cluster_id])
            cluster2value[cluster_id] = value
        return

    cluster2value[cluster_id] = 0
    for child_cluster_id in tree[cluster_id]:
        if child_cluster_id in cluster2value:
            continue
        dfs(tree, child_cluster_id, cluster2value, cluster2nodes, label)
        cluster2value[cluster_id] += cluster2value[child_cluster_id]
    cluster2value[cluster_id] /= len(tree[cluster_id])


def dfs_entrance(cluster2value, nodes_list, tree, cluster2nodes, label):
    for response_id, nodes in enumerate(nodes_list):
        for node in nodes:
            # node_unique_id = get_node_unique_id(response_id, node)
            # cluster_id = node2cluster[node_unique_id]
            cluster_id = node["cluster"]
            if cluster_id not in cluster2value:
                dfs(tree, cluster_id, cluster2value, cluster2nodes, label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--embedding_path", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--included_types", type=str, nargs="+", default=("Thought", "Observation"))
    args = parser.parse_args()

    data = json.load(open(args.input_file, "r"))

    node_ids = []
    sample2node_offset = {}
    unique_id2node = {}
    for j, item in tqdm(enumerate(data)):
        sample2node_offset[j] = [len(node_ids), -1]
        for i in range(len(item["response"])):
            chain_nodes = item["nodes"][i]

            for k in range(len(chain_nodes)):
                if k == len(chain_nodes) - 1:
                    chain_nodes[k]["is_leaf"] = True
                else:
                    chain_nodes[k]["is_leaf"] = False

            # chain_node_ids = [get_node_unique_id(i, node) for node in chain_nodes]
            chain_node_ids = []
            for node in chain_nodes:
                node_id = get_node_unique_id(i, node)
                unique_id2node[node_id] = node
                chain_node_ids.append(node_id)

            node_ids.extend(chain_node_ids)

        sample2node_offset[j][1] = len(node_ids)

    embeddings = np.load(args.embedding_path)

    for i, item in tqdm(enumerate(data), total=len(data)):
        sample_offset = sample2node_offset[i]
        sample_embeddings = embeddings[sample_offset[0]: sample_offset[1]]

        sample_node_ids = node_ids[sample_offset[0]: sample_offset[1]]

        node2cluster = node_clustering(sample_embeddings, sample_node_ids, unique_id2node, threshold=args.threshold, included_types=args.included_types)
        cluster2nodes = collections.defaultdict(list)
        trees = collections.defaultdict(list)
        for j in range(len(item["response"])):
            chain_nodes = item["nodes"][j]

            for k, node in enumerate(chain_nodes):
                node["cluster"] = node2cluster[get_node_unique_id(j, node)]
                cluster2nodes[node["cluster"]].append(node)
                if k > 0:
                    trees[chain_nodes[k - 1]["cluster"]].append(node["cluster"])

            for k, node in enumerate(chain_nodes):
                cluster_id = node["cluster"]
                if cluster_id not in trees and k != len(chain_nodes) - 1:
                    assert node["is_leaf"], f"Warning: {node} is not a leaf node."
                    print(f"Warning: {node} not in trees")

        trees = {k: list(set(v)) for k, v in trees.items() if len(v) > 0}

        cluster2value = {}
        dfs_entrance(cluster2value, item["nodes"], trees, cluster2nodes, item["label"])

        for j in range(len(item["response"])):
            chain_nodes = item["nodes"][j]
            for k, node in enumerate(chain_nodes):
                node["value"] = cluster2value[node["cluster"]]

        item["cluster2nodes"] = cluster2nodes

    json.dump(data, open(args.output_file, "w"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
