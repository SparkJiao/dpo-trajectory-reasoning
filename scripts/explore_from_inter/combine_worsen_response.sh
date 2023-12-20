data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states


# Worsen response to DPO pair

python scripts/construct_dpo_data_via_worsen_response.py \
  --input_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample5.inter_ver2.1.rs0.4.r0.2.?-4.?-of-4.modify_worse.1shot.mistral-7b.json" \
  --original_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.1.rs0.4.r0.2.?-of-4.json" \
  --is_inter_states  \
  --output_file "$data_dir/worsen/logiqav2-train.react.v1.0.0shot.sample5.inter_ver2.1.rs0.4.r0.2.modify_worse.1shot.mistral-7b.dpo.json"

python scripts/construct_dpo_data_via_worsen_response.py \
  --input_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample5.inter_ver2.1.rs0.4.r0.2.?-4.?-of-4.modify_worse.1shot.mistral-7b.json" \
  --original_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.1.rs0.4.r0.2.?-of-4.json" \
  --is_inter_states  \
  --output_file "$data_dir/worsen/logiqav2-train.react.v1.0.0shot.sample5.inter_ver2.1.rs0.4.r0.2.modify_worse.1shot.mistral-7b.dpo.w_wrong.json" \
  --keep_wrong

