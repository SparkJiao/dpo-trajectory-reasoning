sft_model_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/
fix_hack_data_dir=$sft_model_dir/fix_hack_data_dir/

python scripts/construct_dpo_data_from_react_response.py \
  --input_file $sft_model_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.json \
  --output_file $fix_hack_data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_dpo_pair.json

python scripts/split_train_dev.py \
  --input_file $fix_hack_data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_dpo_pair.json