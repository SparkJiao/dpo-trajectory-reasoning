import argparse
import os
from collections import OrderedDict
from glob import glob

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import LlamaForCausalLM, LlamaConfig
from accelerate import init_empty_weights


# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    """

    """
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def split_weights(state_dict: OrderedDict, tp_size: int):
    new_state_dicts = [OrderedDict() for _ in range(tp_size)]

    split_lists = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight",
                   "embed_tokens.weight", "lm_head.weight"]

    for k, v in state_dict.items():
        if any(k.endswith(name) for name in split_lists):
            tensor = list(torch.chunk(v, tp_size, dim=0))
            for i, t in enumerate(tensor):
                new_state_dicts[i][k] = t.detach().clone()
        elif k.endswith("o_proj.weight") or k.endswith("down_proj.weight"):
            tensor = list(torch.chunk(v, tp_size, dim=1))
            for i, t in enumerate(tensor):
                new_state_dicts[i][k] = t.detach().clone()
        else:
            for i in range(tp_size):
                new_state_dicts[i][k] = v.detach().clone()

    return new_state_dicts


def merge_weights(state_dicts: list, merge_avg: bool = False):
    merged_state_dict = OrderedDict()

    col_splits = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight",
                  "embed_tokens.weight", "lm_head.weight"]
    row_splits = ["o_proj.weight", "down_proj.weight"]

    for k in state_dicts[0].keys():
        if any(k.endswith(name) for name in col_splits):
            merged_state_dict[k] = torch.cat([state_dict[k] for state_dict in state_dicts], dim=0)
        elif any(k.endswith(name) for name in row_splits):
            merged_state_dict[k] = torch.cat([state_dict[k] for state_dict in state_dicts], dim=1)
        else:
            merged_state_dict[k] = state_dicts[0][k]
            if merge_avg:
                for i in range(1, len(state_dicts)):
                    merged_state_dict[k] += state_dicts[i][k]
                merged_state_dict[k] /= len(state_dicts)

    print(merged_state_dict.keys())
    return merged_state_dict


def write_model(input_base_path, tp_size: int):
    model = LlamaForCausalLM.from_pretrained(input_base_path, torch_dtype="auto", device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(input_base_path)

    new_state_dicts = split_weights(model.state_dict(), tp_size)
    for i, state_dict in enumerate(new_state_dicts):
        output_folder = os.path.join(input_base_path, f"mp_{i}-of-{tp_size}")
        os.makedirs(output_folder, exist_ok=True)
        model.save_pretrained(output_folder, state_dict=state_dict, safe_serialization=False)
        tokenizer.save_pretrained(output_folder)
        print(f"Model saved to {output_folder}")


def merge_model(input_base_path, tp_size: int, merge_avg: bool = False):
    config = LlamaConfig.from_pretrained(os.path.join(input_base_path, f"mp_0-of-{tp_size}"))
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(input_base_path)

    state_dicts = []
    for folder in glob(os.path.join(input_base_path, f"mp_*-of-{tp_size}")):
        print(folder)
        weights = {}
        files = glob(os.path.join(folder, "*.safetensors"))
        if len(files):
            for weight_path in files:
                with safe_open(weight_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        weights[key] = f.get_tensor(key).clone()
        else:
            for weight_path in glob(os.path.join(folder, "*.bin")):
                weights.update(torch.load(weight_path, map_location="cpu"))
        state_dicts.append(weights)

    merged_state_dict = merge_weights(state_dicts, merge_avg)
    if merge_avg:
        output_folder = os.path.join(input_base_path, "merged_avg")
        model.save_pretrained(output_folder, state_dict=merged_state_dict, safe_serialization=False)
        tokenizer.save_pretrained(output_folder)
        print(f"Model saved to {output_folder}")
    else:
        model.save_pretrained(input_base_path, state_dict=merged_state_dict, safe_serialization=False)
        # tokenizer.save_pretrained(input_base_path)
        print(f"Model saved to {input_base_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument("--tp_size", help="Tensor model parallel size.", default=2, type=int)
    parser.add_argument("--do_split", help="Split the model into shards.", action="store_true", default=False)
    parser.add_argument("--merge_avg", help="Merge the model shards using average.", action="store_true")
    args = parser.parse_args()
    if args.do_split:
        write_model(
            input_base_path=args.input_dir,
            tp_size=args.tp_size,
        )
    else:
        merge_model(
            input_base_path=args.input_dir,
            tp_size=args.tp_size,
            merge_avg=args.merge_avg,
        )


if __name__ == "__main__":
    main()
