data_dir=experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.A100.w2.v1.0/checkpoint-2400


best_of=10
pos_margin=0.7
max_neg_num=10
index="(2,3)"
reduction="product"
reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w4.v1.0.s44/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.prm_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.${reduction}.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction


# ==================================== Debug


#index="(2,3)"
#reduction="product"
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w4.v1.0.s44/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_gpt_reclor_debug_v1.0_cp800_${index}_${reduction}.json" --reduction $reduction --prob_labels ${index}
