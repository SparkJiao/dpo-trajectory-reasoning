sft_model_dir=experiments/llama2.7b.chat.mixtral.dpo-sft.A100.40.w8.v1.0/checkpoint-1200
fix_hack_data_dir=$sft_model_dir/fix_hack_data_dir/

python scripts/construct_dpo_data_from_react_response.py \
  --input_file $sft_model_dir/reclor.react.train.0shot.sample10.v1.0.json \
  --output_file $fix_hack_data_dir/reclor.react.train.0shot.sample10.v1.0.clean_dpo_pair.json

python scripts/split_train_dev.py \
  --input_file $fix_hack_data_dir/reclor.react.train.0shot.sample10.v1.0.clean_dpo_pair.json