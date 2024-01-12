data_dir=experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.A100.w2.v1.0/checkpoint-2400


#best_of=10
##pos_margin=0.7
##pos_margin=0.75
##pos_margin=0.8
##pos_margin=0.85
##pos_margin=0.6
#pos_margin=0.4
#max_neg_num=10
#index="(2,3)"
#reduction="product"
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w8.v1.3.s42/train.reclor.rewards.raw_trajectory.product.v1.1/test-checkpoint-400/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.1.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.prm_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.1.${index}.pair.${reduction}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction


#best_of=10
##pos_margin=0.7
##pos_margin=0.75
##pos_margin=0.8
##pos_margin=0.85
##pos_margin=0.6
#pos_margin=0.2
#max_neg_num=10
#index="(1,2,3)"
##reduction="product"
#reduction="min"
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w8.v1.3.s42/train.reclor.rewards.raw_trajectory.product.v1.0/test-checkpoint-400/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.prm_v13_400_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.${reduction}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction


###################################################################
data_dir=experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.H100.w4.v2.0/checkpoint-1200/

best_of=10
#pos_margin=0.7
#pos_margin=0.75
#pos_margin=0.8
#pos_margin=0.85
#pos_margin=0.6
pos_margin=0.4
max_neg_num=10
index="(2,3)"
reduction="product"
reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w8.v2.0.s42/train.reclor.rewards.raw_trajectory.product.v1.1/test-checkpoint-400/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.1.py \
  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.prm_v20_cp400_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.1.${index}.pair.${reduction}.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction



# Positive pairs only.

#best_of=10
##pos_margin=0.7
##pos_margin=0.75
##pos_margin=0.8
##pos_margin=0.85
##pos_margin=0.6
#pos_margin=0.3
#max_neg_num=10
#index="(1,2,3)"
##reduction="product"
#reduction="min"
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w8.v1.3.s42/train.reclor.rewards.raw_trajectory.product.v1.0/test-checkpoint-400/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2_pos_only.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.prm_v13_400_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.${reduction}.pos_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction

# ==================== Add step num constraints.
#best_of=10
#pos_margin=0.6
#max_neg_num=10
#index="(2,3)"
#reduction="product"
#min_pos_step=7
#min_neg_step=7
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w4.v1.0.s44/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.prm_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.step${min_pos_step}-${min_neg_step}.v2.2.${index}.pair.${reduction}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction \
#  --min_pos_step $min_pos_step --min_neg_step $min_neg_step


#best_of=10
#pos_margin=0.2
#max_neg_num=10
#index="(1,2,3)"
#reduction="min"
#min_pos_step=4
#min_neg_step=-1
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w8.v1.3.s42/train.reclor.rewards.raw_trajectory.product.v1.0/test-checkpoint-400/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.prm_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.step${min_pos_step}-${min_neg_step}.v2.2.${index}.pair.${reduction}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction \
#  --min_pos_step $min_pos_step --min_neg_step $min_neg_step


# ==================================== Debug

#
#index="(2,3)"
#reduction="product"
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w8.v2.0.s42/train.reclor.rewards.raw_trajectory.product.v1.1/test-checkpoint-400/eval_predictions_rank0.json"
#data_dir=experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.H100.w4.v2.0/checkpoint-1200/
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_gpt_reclor_debug_v2.0_cp400_${index}_${reduction}.no_action.json" --reduction $reduction --prob_labels ${index} --remove_action


# ============================================================


## Development set
#index="(1,2,3)"
#reduction="min"
#reward_file="experiments/llama2.7b.chat.reclor.gpt351106.prm.fix_hack.H100.w8.v2.0.s42/train.reclor.rewards.raw_trajectory.product.v1.1/checkpoint-400/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/reclor.react.dev.0shot.sample10.tem0.7.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_gpt_reclor_dev_debug_v1.4_cp800_${index}_${reduction}.json" --reduction $reduction --prob_labels ${index} --remove_last
