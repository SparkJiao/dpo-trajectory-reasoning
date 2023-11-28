data_dir=experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600
#diff=3.0
#diff=2.6
#diff=2.1
diff=3.0
#decay=0.9
#decay=0.8
#decay=0.95
#decay=0.9
decay=1.0

#python scripts/sample_react_inter_states_v2.0.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.json \
#  --output_file $data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_inter_ver2.0.rs0.2.r0.6.json \
#  --split_num 20 --ratio_s 0.2 --ratio 0.6

python scripts/process_inter_response_v2.0.py --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.inter.ver2.0.rs0.2.r0.6.split-*.sample3.json" \
  --output_file "$data_dir/value-ver2.0/logiqav2-train.full.qa.react.v1.0.0shot.inter.ver2.0.rs0.2.r0.6.sample3.diff$diff.decay$decay.value.json" \
  --diff $diff --decay $decay --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_inter_ver2.0.rs0.2.r0.6.*-of-20.json"
