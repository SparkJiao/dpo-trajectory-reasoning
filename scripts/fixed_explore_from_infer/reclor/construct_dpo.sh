#sft_model_dir=experiments/llama2.7b.chat.mixtral.dpo-sft.A100.40.w8.v1.0/checkpoint-1200
#fix_hack_data_dir=$sft_model_dir/fix_hack_data_dir/
#
#python scripts/construct_dpo_data_from_react_response.py \
#  --input_file $sft_model_dir/reclor.react.train.0shot.sample10.v1.0.json \
#  --output_file $fix_hack_data_dir/reclor.react.train.0shot.sample10.v1.0.clean_dpo_pair.json
#
#python scripts/split_train_dev.py \
#  --input_file $fix_hack_data_dir/reclor.react.train.0shot.sample10.v1.0.clean_dpo_pair.json


# ====================================
sft_model_dir=experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.A100.w2.v1.0/checkpoint-2400/
raw_data_file=$sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.json
cleaned_data=$sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json

python scripts/construct_dpo_data_from_react_response_v1.1.py \
  --input_file  $raw_data_file \
  --output_file $sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.clean_dpo_pair_v1.1.no_duplicate.json
# --output_file $sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.clean_dpo_pair_v1.1.json

rs=0.2
r=0.3
inter_file=$sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.clean_inter_ver2.2.rs${rs}.r${r}.json
python scripts/sample_react_inter_states_v2.2.py \
  --input_file $cleaned_data \
  --output_file $inter_file

python scripts/split_train_dev.py \
  --input_file $inter_file
