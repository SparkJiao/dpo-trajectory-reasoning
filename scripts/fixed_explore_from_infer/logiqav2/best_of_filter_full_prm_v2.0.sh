data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states

best_of=10
pos_margin=0.5
max_neg_num=10
index="(1,)"
reduction="product"
up_sampling=1
rm_step=1000
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.H100.w4.v2.0.s42/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/../fix_hack_data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v20_cp${rm_step}_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.${reduction}.up${up_sampling}.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index --reduction ${reduction} --up_sampling ${up_sampling}




# =============================== Debug
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-1600/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/llama2.7b.chat.logiqav2.70b-distil.step.dpo.fix_hack.H100.w4.v1.0.s44.checkpoint-1200.logiqav2.react.train.0shot.sample5.v1.0.cleaned.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_debug_prm_v2_0_fix_hack_cp800_${reduction}_${index}.remove_last.json" --reduction ${reduction} --prob_labels ${index} --remove_last
