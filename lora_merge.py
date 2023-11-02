import argparse
import torch

from peft import PeftModel
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, default="gpt2")
parser.add_argument("--lora_path", type=str, default="gpt2")
parser.add_argument("--output_dir", type=str, default="output")
args = parser.parse_args()

print(f"Loading base model from {args.base_model_path}...")
# model = AutoModelForCausalLM.from_pretrained(args.base_model_path, device_map={"": "cpu"}, low_cpu_mem_usage=True, torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(args.base_model_path)
print(f"Loading lora model from {args.lora_path}...")
# model = PeftModel.from_pretrained(model, args.lora_path, device_map={"": "cpu"}, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, args.lora_path)
print("Merging...")
model = model.merge_and_unload()
print(f"Saving to {args.output_dir}...")
model.save_pretrained(args.output_dir)
