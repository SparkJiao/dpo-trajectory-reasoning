
sft_model_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/


reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.fix_hack.A100.w4.v1.2.s42/sft.dev.n5.tem1.0.reclor.rewards.raw_trajectory.product.v1.1/test-checkpoint-800/eval_predictions_rank0.json"
#margin=0.5
index="(1,2,3)"
reduction="product"
python scripts/combine_reward_debug_v1.0.py \
  --input_file "${sft_model_dir}/logiqav2.dev.react.n5.tem1.0.v1.0.0shot.json" \
  --reward_file $reward_file \
  --output_file "./debug.json" --reduction ${reduction} --prob_labels ${index}


#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.orm.fix_hack.A100.40.w4.v1.2.s42/sft.dev.n5.tem1.0.rewards.raw_trajectory.product.v1.0/test-checkpoint-400/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "${sft_model_dir}/logiqav2.dev.react.n5.tem1.0.v1.0.0shot.json" \
#  --reward_file $reward_file \
#  --output_file "./debug.json"  --prob_labels "(1,)"  --orm

