data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states

rm_step=800
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-${rm_step}/eval_predictions_rank0.json"

python scripts/check_rewards_v1.0.py \
  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --reward_file $reward_file \
  --step_cutoff 50
