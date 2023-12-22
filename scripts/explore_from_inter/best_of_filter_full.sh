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

pos_margin=0.5
#pos_margin=0.3
max_neg_num=1
python scripts/best_of_filter_by_reward_v1.2.py \
  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w8.v1.1/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-600/eval_predictions_rank0.json" \
  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_v11_best_of_${best_of}.neg${max_neg_num}.pos${pos_margin}.v1.2.pair.full_only.json" \
  --best_of $best_of --contrastive --max_neg_num $max_neg_num --pos_margin $pos_margin

#max_neg_num=6
#python scripts/best_of_filter_by_reward_v1.1.py \
#  --input_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.prm.H100.w8.v1.1/train.rewards.raw_trajectory.product.v1.0/test-checkpoint-600/eval_predictions_rank0.json" \
#  --output_file "$data_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_full_v11_best_of_${best_of}.neg${max_neg_num}.v1.1.pair.full_only.json" \
#  --best_of $best_of --contrastive --max_neg_num $max_neg_num