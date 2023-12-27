data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states

best_of=3
#max_neg_num=6
max_neg_num=4
#pos_margin=2.0
pos_margin=1.5

#python scripts/best_of_filter_by_reward_v1.0.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.rm_v10_best_of_${best_of}.pair.full_only.json" \
#  --best_of $best_of --contrastive
#
#python scripts/best_of_filter_by_reward_v1.0.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.full.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.rm_full_v10_best_of_${best_of}.pair.full_only.json" \
#  --best_of $best_of --contrastive

#python scripts/best_of_filter_by_reward_v1.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.rm_v10_best_of_${best_of}.neg${max_neg_num}.v1.1.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num

#python scripts/best_of_filter_by_reward_v1.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.full.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.rm_full_v10_best_of_${best_of}.neg${max_neg_num}.v1.1.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num


#python scripts/best_of_filter_by_reward_v1.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.rm_v10_best_of_${best_of}.neg${max_neg_num}.v1.1.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num
#

#python scripts/best_of_filter_by_reward_v1.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.rm_v10_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v1.2.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --pos_margin $pos_margin


#python scripts/best_of_filter_by_reward_v1.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.full.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.raw_response.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.rm_full_v10_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v1.2.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --pos_margin $pos_margin

#pos_margin=0.5
#pos_margin=0.3
#max_neg_num=1
#python scripts/best_of_filter_by_reward_v1.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w8.v1.1/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-600/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v11_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v1.2.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --pos_margin $pos_margin


#max_neg_num=6
#python scripts/best_of_filter_by_reward_v1.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w8.v1.1/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-600/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v11_best_of_${best_of}.neg${max_neg_num}.v1.1.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num


# Fix bug since 2023/12/23.
#best_of=3
#max_neg_num=6
#python scripts/best_of_filter_by_reward_v1.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.1.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-600/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v11_best_of_${best_of}.neg${max_neg_num}.v1.1.pair.full_only.fix.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num
#
#best_of=3
#pos_margin=0.5
#max_neg_num=4
#python scripts/best_of_filter_by_reward_v1.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.1.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-600/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v11_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v1.2.pair.full_only.fix.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --pos_margin $pos_margin
#
#
#best_of=3
#max_neg_num=6
#inter_best_of=3
#inter_margin=0.3
#inter_max_neg_num=4
#python scripts/best_of_filter_by_reward_v1.3.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.1.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-600/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v11_best_of_${best_of}.neg${max_neg_num}.in${inter_best_of}.in_neg${inter_max_neg_num}.in_m${inter_margin}.v1.3.pair.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --inter_best_of $inter_best_of --inter_max_neg_num $inter_max_neg_num --inter_margin $inter_margin


# Process-Reward model v1.2 fix
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#best_of=3
#max_neg_num=6
#python scripts/best_of_filter_by_reward_v1.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v12_best_of_${best_of}.neg${max_neg_num}.v1.1.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num
#
#best_of=3
#pos_margin=0.3
#max_neg_num=4
#python scripts/best_of_filter_by_reward_v1.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v1.2.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --pos_margin $pos_margin


#best_of=3
#max_neg_num=4
#inter_best_of=3
#inter_margin=0.3
#inter_max_neg_num=4
#python scripts/best_of_filter_by_reward_v1.3.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v12_best_of_${best_of}.neg${max_neg_num}.in${inter_best_of}.in_neg${inter_max_neg_num}.in_m${inter_margin}.v1.3.pair.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --inter_best_of $inter_best_of --inter_max_neg_num $inter_max_neg_num --inter_margin $inter_margin

# ++++++++++++++++++++++++++++++++++++++++++ Logs here.
#collected rewards 108096
#duplicate responses 15280
#Reduced 3089
#Candidates: 12560
#Collected amount of samples with rewards 79727
#Save to experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v12_best_of_3.neg6.v1.1.pair.full_only.json
#collected rewards 108096
#duplicate responses 15280
#Reduced 3089
#Positive pairs 31178
#Candidates: 12560
#Collected amount of samples with rewards 94782
#Save to experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_best_of_3.neg4.pos0.3.v1.2.pair.full_only.json
#collected rewards 1494964
#duplicate responses 15280
#Reduced 3089
#Candidates: 12560
#Inter: 68605
#Missed full: 0
#Missed partial: 0
#Collected amount of samples with rewards 132209
#Save to experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v12_best_of_3.neg4.in3.in_neg4.in_m0.3.v1.3.pair.json

#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#best_of=3
#pos_margin=0.7
#max_neg_num=6
#python scripts/best_of_filter_by_reward_v1.2.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v1.2.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num --pos_margin $pos_margin


# Re-compute the reward by using the probability of `label==3` only.
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
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


best_of=10
pos_margin=0.4
max_neg_num=10
index="(3,)"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-1600/eval_predictions_rank0.json"
reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
python scripts/best_of_filter_by_reward_v2.2.py \
  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --reward_file $reward_file \
  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v12_cp800_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v2.2.$index.pair.product.full_only.json" \
  --best_of $best_of --max_neg_num $max_neg_num --pos_margin $pos_margin --prob_labels $index

# =============================== Debug
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-800/eval_predictions_rank0.json"
#reward_file="experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w4.v1.2.s42.fix/train.rewards.raw_trajectory.product.v1.0.fix/test-checkpoint-1600/eval_predictions_rank0.json"
#python scripts/combine_reward_debug_v1.0.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file $reward_file \
#  --output_file "./reward_debug_cp800_min_(1,2,3).json" --reduction min --prob_labels "(1,2,3)"
