data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states

#best_of=3
##max_neg_num=6
#max_neg_num=4
##pos_margin=2.0
#pos_margin=1.5

# Re-compute the reward by using the probability of `label==3` only.
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#best_of=3
#max_neg_num=6
#python scripts/best_of_filter_by_reward_v2.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v12_best_of_${best_of}.neg${max_neg_num}.v2.1.pair.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num
#
#best_of=3
#pos_margin=0.7
#max_neg_num=6
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.pair.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin

#best_of=10
#pos_margin=0.7
#max_neg_num=10
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.pair.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin


#margin=0.6
#python scripts/best_of_filter_by_reward_v2.4.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12.mar${margin}.v2.4.pair.full_only.json" \
#  --margin $margin --reduction "product"

#margin=0.5
#python scripts/best_of_filter_by_reward_v2.4.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12.mar${margin}.v2.4.pair.full_only.json" \
#  --margin $margin --reduction "product"

#margin=0.5
#python scripts/best_of_filter_by_reward_v2.5.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12.mar${margin}.v2.5.(1,2,3,).pair.full_only.json" \
#  --margin $margin --reduction "product" --prob_labels "(1,2,3)"


#best_of=10
#pos_margin=0.7
#max_neg_num=10
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.(1,2,3).pair.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels "(1,2,3)"


#best_of=10
#pos_margin=0.7
#max_neg_num=10
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.(1,2,3).pair.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels "(1,2,3)"


#best_of=10
#pos_margin=0.7
#max_neg_num=10
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-1600/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_cp1600_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.(1,2,3).pair.min.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels "(1,2,3)" --reduction "min"


#best_of=10
#pos_margin=0.5
#max_neg_num=10
#index="(2,3)"
#reduction="product"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.${index}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index

# Reduction as "min" @ 2023/12/29
#best_of=10
#pos_margin=0.25
#max_neg_num=10
#index="(2,3)"
#reduction="min"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.${index}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction}

# ========================================= Adjust `index` and `best_of` ===================== @ 2023/12/30
#best_of=3
#pos_margin=0.5
#max_neg_num=10
#index="(1,2,3)"
#reduction="product"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
##python scripts/best_of_filter_by_reward_v2.2.py \
##  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
##  --reward_file $reward_file \
##  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.${index}.full_only.json" \
##  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction}
#python scripts/best_of_filter_by_reward_v2.6.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.6.$index.pair.${reduction}.${index}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction}


#margin=0.5
#python scripts/best_of_filter_by_reward_v2.5.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12.mar${margin}.v2.4.pair.full_only.json" \
#  --margin $margin --reduction "product" --prob_labels

# ============================== Add up-sampling   @ 2023/12/30
#best_of=10
#pos_margin=0.5
#max_neg_num=10
#index="(2,3)"
#reduction="product"
#up_sampling=3
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}
#
#
#best_of=10
#pos_margin=0.5
#max_neg_num=10
#index="(1,2,3)"
#reduction="product"
#up_sampling=3
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}


best_of=10
pos_margin=0.7
max_neg_num=10
index="(1,2,3)"
reduction="product"
up_sampling=3
rm_step=800
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}

best_of=10
pos_margin=0.7
max_neg_num=10
index="(1,2,3)"
reduction="product"
up_sampling=3
rm_step=1200
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}


# =============================== Ver2.4 @ 2023/12/31
#margin=0.5
#index="(2,3)"
#reduction="product"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.4.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800.margin${margin}.v2.4.$index.pair.${reduction}.full_only.json" \
#  --margin $margin --prob_labels $index --reduction ${reduction}
# To few negative-negative pairs, meaning that there is no difference with ver2.2.


# =============================== Debug
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-1600/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/llama2.7b.chat.logiqav2.70b-distil.step.dpo.fix_hack.H100.w4.v1.0.s44.checkpoint-1200.logiqav2.react.train.0shot.sample5.v1.0.cleaned.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_debug_fix_hack_cp800_${product}_${index}.json" --reduction ${reduction} --prob_labels ${index}
##