############################### Initial seed responses for SFT

#python scripts/construct_dpo_data_from_react_response_v1.1.py \
#  --input_file /export/home2/fangkai/rl-hybrid-engine/api-outputs/gpt35turbo1106/logiqav2.train.react.1shot.gpt35turbo1106.sample10.tem0.7.json \
#  --output_file api-outputs/gpt35turbo1106/reclor.train.react.1shot.gpt35turbo1106.sample10.tem0.7.v1.1.min_step_8.dpo_pair.json \
#  --min_step 8


data_dir=experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.H100.w4.v2.0/checkpoint-1200/
rs=0.2
r=0.5
min_step=8
#python scripts/sample_react_inter_states_v2.3.py \
#  --input_file $data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.sub_dev.1000.json \
#  --output_file $data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.sub_dev.1000.inter_ver2.3.rs${rs}.r${r}.min_step_${min_step}.no_act.json \
#  --ratio_s $rs --ratio $r --min_step $min_step --remove_action --split_num 2
#python scripts/sample_react_inter_states_v2.3.py \
#  --input_file $data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.sub_train.3638.sub_dev.200.json \
#  --output_file $data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.sub_train.3638.sub_dev.200.inter_ver2.3.rs${rs}.r${r}.min_step_${min_step}.no_act.json \
#  --ratio_s $rs --ratio $r --min_step $min_step --remove_action --split_num 2

python scripts/construct_dpo_data_from_react_response_v1.1.py \
  --input_file $data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.json \
  --output_file $data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.min_step_8.dpo_pair.json \
  --min_step 8

####################################################

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

#python scripts/construct_dpo_data_from_react_response_v1.1.py \
#  --input_file  $raw_data_file \
#  --output_file $sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.clean_dpo_pair_v1.1.no_duplicate.json
## --output_file $sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.clean_dpo_pair_v1.1.json
#
#rs=0.2
#r=0.3
#inter_file=$sft_model_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.clean_inter_ver2.2.rs${rs}.r${r}.json
#python scripts/sample_react_inter_states_v2.2.py \
#  --input_file $cleaned_data \
#  --output_file $inter_file
#
#python scripts/split_train_dev.py \
#  --input_file $inter_file

#python scripts/construct_dpo_data_from_react_response_v1.1.py \
#  --input_file ${sft_model_dir}/reclor.react.dev.0shot.sample10.tem0.7.v1.0.json \
#  --output_file ${sft_model_dir}/reclor.react.dev.0shot.sample10.tem0.7.v1.0.clean_dpo_pair_v1.1.no_duplicate.json

#python scripts/construct_dpo_data_from_react_response_v1.1.py \
#  --input_file "${sft_model_dir}/reclor.react.train.0shot.sample20.tem0.7.?-of-2v1.0.json" \
#  --output_file ${sft_model_dir}/reclor.react.train.0shot.sample20.tem0.7.v1.0.clean_dpo_pair_v1.1.no_duplicate.json

#python scripts/construct_dpo_data_from_react_response_v1.1.py \
#  --input_file "${sft_model_dir}/response30merge/reclor.react.train.0shot.resp30merge.tem0.7.v1.0.json" \
#  --output_file ${sft_model_dir}/response30merge/reclor.react.train.0shot.resp30merge.tem0.7.v1.0.clean_dpo_pair_v1.1.no_duplicate.json

#4638
#Clean: 4638 -> 4638
#Average response number: 19.22013799051315
#55534
#278859
#{'invalid': 0, 'missing': 124, 'multiple': 59, 'wrong': 33426}


# =====================================================

#model_dir=experiments/llama2.7b.chat.logiqav2.70b-distil.step.dpo.fix_hack.H100.w4.v1.0.th.s43/checkpoint-2000/
#
#
#python scripts/construct_dpo_data_from_react_response_v1.1.py \
#  --input_file ${model_dir}/reclor.react.train.0shot.sample10.tem0.7.json \
#  --output_file ${model_dir}/reclor.react.train.0shot.sample10.tem0.7.v1.0.clean_dpo_pair_v1.1.no_duplicate.json

