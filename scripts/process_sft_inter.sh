data_dir=experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600

python scripts/sample_react_inter_states.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.json \
  --output_file $data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_inter_ver0.0.json \
  --split_num 10