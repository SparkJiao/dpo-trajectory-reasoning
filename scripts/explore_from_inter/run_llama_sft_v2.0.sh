data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600

ratio_s=0.2
ratio=0.3

#python scripts/sample_react_inter_states_v2.0.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.json \
#  --output_file $data_dir/react-inter-states/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.6.json \
#  --split_num 20 --ratio_s 0.2 --ratio 0.6

python scripts/sample_react_inter_states_v2.0.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.json \
  --output_file $data_dir/react-inter-states/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs${ratio_s}.r${ratio}.json \
  --split_num 20 --ratio_s ${ratio_s} --ratio ${ratio}
