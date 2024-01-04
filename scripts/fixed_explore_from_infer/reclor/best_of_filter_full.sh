data_dir=experiments/llama2.7b.chat.mixtral.dpo-sft.A100.40.w8.v1.0/checkpoint-1200/react-inter-states


#best_of=10
#pos_margin=0.2
#max_neg_num=10
#index="(2,3,4,5)"
##reward_file="experiments/llama2.7b.chat.reclor.mixtral.prm.fix_hack.A40.w8.v1.0.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_cp2400_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.product.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction "product"


# Use min scores instead of product. @ 2023/12/29
#best_of=10
#pos_margin=0.2
#max_neg_num=10
#reduction="min"
#index="(2,3,4,5)"
#reward_file="experiments/llama2.7b.chat.reclor.mixtral.prm.fix_hack.A40.w8.v1.0.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.${reduction}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction


# Use reward model prm v1.2 @ 2024/01/04

best_of=10
pos_margin=0.1
max_neg_num=10
index="(2,3,4,5)"
reduction="product"
reward_file="experiments/llama2.7b.chat.reclor.mixtral.prm.fix_hack.A100.40.w8.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/../fix_hack_data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.${reduction}.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction $reduction

# ===================== Sampling from DPO model.
#best_of=10
#pos_margin=0.3
#max_neg_num=10
#index="(2,3,4,5)"
#reward_file="experiments/llama2.7b.chat.reclor.mixtral.prm.fix_hack.A40.w8.v1.0.s42/llama2.7b.chat.reclor.mixtral.dpo.fix_hack.A100.40.w8.v2.0.s42.checkpoint-400.reclor.react.train.0shot.sample10.v1.0.cleaned.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#data_dir="experiments/llama2.7b.chat.reclor.mixtral.dpo.fix_hack.A100.40.w8.v2.0.s42/checkpoint-400/react-inter-states/"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/reclor.react.train.0shot.sample10.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.cleaned.prm_v12_cp400_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.${index}.pair.product.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels ${index} --reduction "product"

# =============================== Debug
#index="(2,3,4,5)"
#reduction="product"
##reduction="sum"
##reduction="min"
#reward_file="experiments/llama2.7b.chat.reclor.mixtral.prm.fix_hack.A100.40.w8.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
##reward_file="experiments/llama2.7b.chat.reclor.mixtral.prm.fix_hack.A40.w8.v1.0.s42/llama2.7b.chat.reclor.mixtral.dpo.fix_hack.A100.40.w8.v2.0.s42.checkpoint-400.reclor.react.train.0shot.sample10.v1.0.cleaned.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/reclor.train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_reclor_debug_cp800_${index}_${reduction}.json" --reduction $reduction --prob_labels ${index}
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "experiments/llama2.7b.chat.reclor.mixtral.dpo.fix_hack.A100.40.w8.v2.0.s42/checkpoint-400/react-inter-states/reclor.react.train.0shot.sample10.v1.0.cleaned.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_reclor_debug_iter1_cp800_${index}_${reduction}.json" --reduction $reduction --prob_labels ${index}
#