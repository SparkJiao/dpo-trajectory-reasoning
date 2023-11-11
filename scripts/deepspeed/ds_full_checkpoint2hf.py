import argparse
import json
import os
from glob import glob
from pathlib import Path

import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM


def extract_weight(mp_states):
    state_dicts = torch.load(mp_states, map_location="cpu")
    state_dicts = state_dicts["module"]
    return state_dicts


def write_model(input_base_path, mp_states_name, config_dir):
    config = transformers.AutoConfig.from_pretrained(config_dir)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    if os.path.exists(input_base_path):
        checkpoint_dirs = [input_base_path]
    else:
        checkpoint_dirs = glob(input_base_path, recursive=True)
    print(f"Found checkpoints: {checkpoint_dirs}")

    for checkpoint_dir in checkpoint_dirs:
        print(f"Writing checkpoint: {checkpoint_dir}")
        states_file = os.path.join(checkpoint_dir, mp_states_name)
        checkpoint_state_dict = extract_weight(states_file)
        step = checkpoint_dir.split("global_step")[-1]
        save_dir = os.path.join(os.path.dirname(checkpoint_dir), f"checkpoint-{step}")
        print(f"Saving checkpoint to {save_dir}")
        model.save_pretrained(save_dir, state_dict=checkpoint_state_dict, max_shard_size="3GB", safe_serialization=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument("--mp_states_name", type=str, default="mp_rank_00_model_states.pt")
    parser.add_argument(
        "--config_dir",
    )
    args = parser.parse_args()
    write_model(
        input_base_path=args.input_dir,
        mp_states_name=args.mp_states_name,
        config_dir=args.config_dir,
    )


if __name__ == "__main__":
    main()
