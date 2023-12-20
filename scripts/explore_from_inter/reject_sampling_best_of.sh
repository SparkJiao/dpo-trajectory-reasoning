data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states

best_of=1
inter_best_of=1

python scripts/reject_sample_best_of_filter_by_reward_v1.0.py \
  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.best_of_${best_of}_${inter_best_of}.json" \
  --best_of $best_of --inter_best_of $inter_best_of
