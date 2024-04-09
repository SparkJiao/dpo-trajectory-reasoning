sft_dir=experiments/gemma.2b.it.meta_math_distil.H100.w4.v1.0/checkpoint-2000
#input_file="${sft_dir}/meta_math_sub.25k.rap.train.0shot.n10.tem1.0.p0.7.v1.0_clean_fix.json"
input_file="${sft_dir}/meta_math_sub.25k.rap.train.0shot.n20.tem1.0.p0.8.v1.0_clean.json"

best_of=10
pos_margin=0.5
max_neg_num=10
#index="(2,3)"
index="(1,2,3)"
reduction="product"
rm_step=3200
#reward_file="experiments/gemma.2b.it.meta_math_distil.prm.H100.w4.v1.0/meta_math_sub.25k.rap.train.0shot.n10.tem1.0.p0.7.v1.0_clean_fix.rewards.v1.0/test-checkpoint-${rm_step}/eval_predictions_rank0.json"
reward_file="experiments/gemma.2b.it.meta_math_distil.prm.H100.w4.v1.0/meta_math_sub.25k.rap.train.0shot.n20.tem1.0.p0.8.v1.0_clean.rewards.v1.0/test-checkpoint-3200/eval_predictions_rank0.json"

#python scripts/cot/dpo_pair_by_reward_v1.0.py \
#  --input_file $input_file \
#  --reward_file $reward_file \
#  --output_file "$sft_dir/meta_math_sub.25k.rap.train.0shot.n10.tem1.0.p0.7.v1.0_clean_fix.prm_cp${rm_step}_best_of_${best_of}.v1.0.$index.pos$pos_margin.neg$max_neg_num.${reduction}.full_only.json" \
#  --best_of $best_of --prob_labels $index --reduction ${reduction} --max_neg_num $max_neg_num --pos_margin $pos_margin

#python scripts/cot/dpo_pair_by_reward_v1.0.py \
#  --input_file $input_file \
#  --reward_file $reward_file \
#  --output_file "$sft_dir/meta_math_sub.25k.rap.train.0shot.n20.tem1.0.p0.8.v1.0_clean.prm_cp${rm_step}_best_of_${best_of}.v1.0.$index.pos$pos_margin.neg$max_neg_num.${reduction}.full_only.json" \
#  --best_of $best_of --prob_labels $index --reduction ${reduction} --max_neg_num $max_neg_num --pos_margin $pos_margin




# ===============================
math_pos_margin=0.5
gsm_pos_margin=0.5
#python scripts/cot/meta_math_dpo_pair_by_reward_v1.0.py \
#  --input_file $input_file \
#  --reward_file $reward_file \
#  --output_file "$sft_dir/meta_math_sub.25k.rap.train.0shot.n10.tem1.0.p0.7.v1.0_clean_fix.prm_cp${rm_step}_best_of_${best_of}.v1.0.$index.math${math_pos_margin}.gsm${gsm_pos_margin}.neg$max_neg_num.${reduction}.full_only.json" \
#  --best_of $best_of --prob_labels $index --reduction ${reduction} --max_neg_num $max_neg_num \
#  --gsm_pos_margin $gsm_pos_margin --math_pos_margin $math_pos_margin

gsm_reward_file="experiments/gemma.2b.it.meta_math_distil.prm.H100.w4.v1.2.type_gsm.s42/meta_math_sub.gsm8k.25k.rap.train.0shot.n20.tem1.0.p0.8.v1.0_clean.rewards.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
math_reward_file="experiments/gemma.2b.it.meta_math_distil.prm.H100.w4.v1.2.type_math.s42/meta_math_sub.math.25k.rap.train.0shot.n20.tem1.0.p0.8.v1.0_clean.rewards.v1.0/test-checkpoint-800/eval_predictions_rank0.json"
python scripts/cot/meta_math_dpo_pair_by_reward_v1.0.py \
  --input_file $input_file \
  --reward_file $gsm_reward_file \
  --reward_file_2 $math_reward_file \
  --output_file "$sft_dir/meta_math_sub.25k.rap.train.0shot.n20.tem1.0.p0.8.v1.0_clean.prm.gsm_cp800.math_cp800.best_of_${best_of}.v1.0.$index.math${math_pos_margin}.gsm${gsm_pos_margin}.neg$max_neg_num.${reduction}.full_only.json" \
  --best_of $best_of --prob_labels $index --reduction ${reduction} --max_neg_num $max_neg_num \
  --gsm_pos_margin $gsm_pos_margin --math_pos_margin $math_pos_margin
