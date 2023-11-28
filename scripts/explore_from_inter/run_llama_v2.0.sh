data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states
#diff=3.0
diff=2.6
#diff=2.1
#diff=3.0
#decay=0.9
#decay=0.8
decay=0.95
#decay=0.9
#decay=1.0


python scripts/process_inter_response_v2.0.py \
  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
  --diff $diff --decay $decay --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json"
