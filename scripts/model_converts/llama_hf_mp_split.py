import argparse
import json
import os
from glob import glob
from pathlib import Path

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers import AutoTokenizer
from collections import OrderedDict


# permute for sliced rotary
def permute(w, n_heads, dim1, dim2):
    """

    """
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def split_weights(state_dict: OrderedDict, tp_size: int):
    new_state_dicts = [OrderedDict() for _ in range(tp_size)]

    split_lists = ["q_proj.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight",
                   "embed_tokens.weight", "lm_head.weight"]

    for k, v in state_dict.items():
        if any(k in name for name in split_lists):
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


def write_model(input_base_path, tp_size: int):
    model = LlamaForCausalLM.from_pretrained(input_base_path)
    tokenizer = AutoTokenizer.from_pretrained(input_base_path)

    new_state_dicts = split_weights(model.state_dict(), tp_size)
    for i, state_dict in enumerate(new_state_dicts):
        output_folder = os.path.join(input_base_path, f"mp_{i}-of-{tp_size}")
        os.makedirs(output_folder, exist_ok=True)
        model.save_pretrained(output_folder, state_dict=state_dict)
        tokenizer.save_pretrained(output_folder)
        print(f"Model saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument("--tp_size", help="Tensor model parallel size.", default=2, type=int)
    parser.add_argument("--do_split", help="Split the model into shards.", action="store_true")
    args = parser.parse_args()
    if args.do_split:
        write_model(
            input_base_path=args.input_dir,
            tp_size=args.tp_size,
        )


if __name__ == "__main__":
    main()
