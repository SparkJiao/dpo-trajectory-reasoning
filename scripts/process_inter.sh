python scripts/process_inter_response.py \
  --input_file "experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.0shot.inter_completion.split-*.json" \
  --output_file logiqav2-train.full.qa.react.v1.0.0shot.inter_completion.pair_diff3.json \
  --diff 3 \
  --inter_state_file "experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_inter_ver0.0.*-of-10.json"