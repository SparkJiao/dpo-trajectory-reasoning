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

best_of=10
#pos_margin=0.5
pos_margin=0.7
#pos_margin=0.5
#pos_margin=0.3
max_neg_num=10
index="(1,2,3)"
#index="(2,3)"
reduction="product"
up_sampling=1
rm_step=800
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w8.v1.0.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/logiqav2.train.react.v1.0.0shot.sample10.prm_hack_fix_v10_iter1_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
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
margin=0.5
index="(2,3)"
reduction="product"

#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-1600/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/llama2.7b.chat.logiqav2.70b-distil.step.dpo.fix_hack.H100.w4.v1.0.s44.checkpoint-1200.logiqav2.react.train.0shot.sample5.v1.0.cleaned.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w8.v1.0.iter1.s42/train.logiqav2.rewards.raw_trajectory.product.step-dpo-v1.0.v1.1/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/logiqav2.react.train.0shot.sample10.tem1.0.v1.0.cleaned.min_step_0.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_debug_fix_hack_cp800_iter1_${reduction}_${index}.json" --reduction ${reduction} --prob_labels ${index}
