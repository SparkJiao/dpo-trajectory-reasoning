data_dir=experiments/llama2.7b.chat.mixtral.dpo-sft.A100.40.w8.v1.0/checkpoint-1200/react-inter-states


best_of=10
pos_margin=0.7
max_neg_num=10
index="(1,2,3,4,5)"
reward_file="experiments/llama2.7b.chat.reclor.mixtral-distil.prm.A100.40.w8.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-2400/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_cp2400_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.product.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction "product"


# =============================== Debug
#index="(1,2,3,4,5)"
#reward_file="experiments/llama2.7b.chat.reclor.mixtral-distil.prm.A100.40.w8.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-2400/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_reclor_debug_cp2400_${index}.json" --reduction product --prob_labels ${index}
