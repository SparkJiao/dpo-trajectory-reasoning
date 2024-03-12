data_dir=experiments/llama2.7b.chat.logiqav2.70b-distil.step.dpo.fix_hack.H100.w4.v1.0.th.s42/checkpoint-400/

#best_of=10
##pos_margin=0.5
##pos_margin=0.7
#pos_margin=0.8
##pos_margin=0.3
#max_neg_num=10
##index="(1,2,3)"
#index="(2,3)"
#reduction="product"
#up_sampling=1
#rm_step=800
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v10_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}


# ===================================== self-sampled distribution based reward model ===========================================

#best_of=10
##pos_margin=0.5
##pos_margin=0.7
#pos_margin=0.8
##pos_margin=0.5
##pos_margin=0.3
#max_neg_num=10
#index="(1,2,3)"
##index="(2,3)"
#reduction="product"
#up_sampling=1
#rm_step=800
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w8.v1.0.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v10_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}

# ================================== Remove action

#best_of=10
##pos_margin=0.5
##pos_margin=0.7
#pos_margin=0.8
##pos_margin=0.75
##pos_margin=0.5
##pos_margin=0.3
#max_neg_num=10
#index="(1,2,3)"
##index="(2,3)"
#reduction="min"
##reduction='min'
#up_sampling=1
#rm_step=800
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w8.v1.0.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.1.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v10_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.1.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}
#
## Add step length constraint to only positive-positive pairs.

#best_of=10
#pos_margin=0.6
##pos_margin=0.7
##pos_margin=0.8
##pos_margin=0.75
##pos_margin=0.5
##pos_margin=0.3
#max_neg_num=10
#index="(1,2,3)"
##index="(2,3)"
#reduction="product"
#up_sampling=1
#rm_step=800
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w8.v1.0.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.1.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v10_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.1.$index.pair.${reduction}.up${up_sampling}.step8-8.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling} \
#  --min_neg_step 8 --min_pos_step 8


# ========================================================= W/ action

#best_of=10
##pos_margin=0.5
##pos_margin=0.7
#pos_margin=0.8
##pos_margin=0.85
##pos_margin=0.9
##pos_margin=0.5
##pos_margin=0.3
#max_neg_num=10
##index="(1,2,3)"
##index="(2,3)"
#index="(3,)"
#reduction="product"
#up_sampling=1
#rm_step=1200
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w8.v1.1.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v11_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}


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


#best_of=10
##pos_margin=0.5
##pos_margin=0.7
#pos_margin=0.8
##pos_margin=0.85
##pos_margin=0.9
##pos_margin=0.5
##pos_margin=0.3
#max_neg_num=10
#index="(2,3)"
##index="(2,3)"
##index="(3,)"
#reduction="product"
#up_sampling=1
#rm_step=600
##reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w2.v1.2.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-$rm_step/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w4.v1.3.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-$rm_step/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v13_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}

# ============================= Exclude the training set for reward model

#best_of=10
#pos_margin=0.5
#max_neg_num=10
#index="(2,3)"
#reduction="product"
#up_sampling=1
#rm_step=600
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w4.v1.3.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-$rm_step/eval_predictions_rank0.json"
#python scripts/best_of_filter_by_reward_v2.2.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v13_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.ex_rm_tr.json" \
#  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling} \
#  --exclude_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned_inter_ver2.3.rs0.1.r0.5.re0.8.min_step_0.sub_dev.1000.json"
#


# ================================== Replay based reward model

best_of=10
#pos_margin=0.8
pos_margin=0.9
max_neg_num=10
index="(2,3)"
reduction="product"
up_sampling=1
rm_step=600
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w4.v1.1.iter1.replay0.1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_replay_v11_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}


# =============================== Debug
#margin=0.5
#index="(1,2,3)"
index="(1,2,3)"
reduction="product"

#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-1600/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/llama2.7b.chat.logiqav2.70b-distil.step.dpo.fix_hack.H100.w4.v1.0.s44.checkpoint-1200.logiqav2.react.train.0shot.sample5.v1.0.cleaned.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w8.v1.0.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-800/eval_predictions_rank0.json"
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w4.v1.3.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-600/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_debug_prm_v13_cp600_iter1_${reduction}_${index}.json" --reduction ${reduction} --prob_labels ${index}
