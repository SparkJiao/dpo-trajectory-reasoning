data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600
#data_dir=experiments/llama2.7b.chat.logiqav2.70b-distil.dpo.H100.w4.v1.0/checkpoint-1600

#ratio_s=0.2
#ratio=0.3
ratio_s=0.4
ratio=0.2

#python scripts/sample_react_inter_states_v2.0.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.json \
#  --output_file $data_dir/react-inter-states/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.6.json \
#  --split_num 20 --ratio_s 0.2 --ratio 0.6

#python scripts/sample_react_inter_states_v2.0.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.json \
#  --output_file $data_dir/react-inter-states/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs${ratio_s}.r${ratio}.json \
#  --split_num 20 --ratio_s ${ratio_s} --ratio ${ratio}

#python scripts/sample_react_inter_states_v2.0.py --input_file $data_dir/logiqav2-train.react.sample5.v1.0.0shot.json \
#  --output_file $data_dir/react-inter-states/logiqav2-train.react.v1.0.0shot.sample5.clean_inter_ver2.0.rs${ratio_s}.r${ratio}.json \
#  --split_num 10 --ratio_s ${ratio_s} --ratio ${ratio}

#python scripts/sample_react_inter_states_v2.1.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.json \
#  --output_file $data_dir/react-inter-states/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.1.rs${ratio_s}.r${ratio}.json \
#  --split_num 4 --ratio_s ${ratio_s} --ratio ${ratio}



# ================================= ReClor
data_dir="experiments/llama2.7b.chat.mixtral.dpo-sft.A100.40.w8.v1.0/checkpoint-1200"
ratio_s=0.2
ratio=0.3
python scripts/sample_react_inter_states_v2.0.py --input_file $data_dir/reclor.react.train.0shot.sample10.v1.0.json \
  --output_file $data_dir/react-inter-states/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs${ratio_s}.r${ratio}.json \
  --split_num 1 --ratio_s ${ratio_s} --ratio ${ratio}


