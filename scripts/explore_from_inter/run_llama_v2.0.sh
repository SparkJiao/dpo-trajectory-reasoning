data_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/react-inter-states
#diff=3.0
#diff=2.6
#diff=2.1
#diff=3.0
diff=2.0
#decay=0.9
#decay=0.8
#decay=0.95
#decay=0.9
#decay=1.0
#step_ratio_diff=0.4
step_ratio_diff=0.3
#negative_threshold=0.0
#negative_threshold=-1.0
#negative_threshold=0.0
negative_threshold=1.0
step_id_clip="[5,30]"
#negative_num=2
negative_num=1
seed=42

#chosen_l=1.5
#chosen_l=1.0
#reject_r=-2
#chosen_l=-0.5
#reject_r=-2
#chosen_l=-0.5
#reject_r=-1.5
chosen_r=5.0
reject_l=0.0

#margin=1.5
margin=1.2



#python scripts/process_inter_response_v2.0.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --diff $diff --decay $decay --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json"


#python scripts/process_inter_response_v2.1.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.1.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --diff $diff --decay $decay --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json"

#python scripts/process_inter_response_v2.3.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.3.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --diff $diff --step_ratio_diff $step_ratio_diff \


#python scripts/process_inter_response_v2.3_by_rewards.py \
#  --input_file experiments/llama2.7b.chat.logiqav2.70b-distil.rm.A100.w4.v4.0/train.rewards.v1.0/test-checkpoint-600/eval_predictions_rank0.json \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.3.rs0.2.r0.3.sample3.$chosen_r.$reject_l.step_r_diff$step_ratio_diff.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json"


#python scripts/process_inter_response_v2.3_by_rewards.py \
#  --input_file experiments/llama2.7b.chat.logiqav2.70b-distil.rm.A100.w4.v4.0/train.rewards.v1.0/test-checkpoint-600/eval_predictions_rank0.json \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.3.rs0.2.r0.3.sample3.$chosen_r.$reject_l.step_r_diff$step_ratio_diff.exclude_full.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --chosen_r $chosen_r --reject_l $reject_l --step_ratio_diff $step_ratio_diff --exclude_full

#python scripts/process_inter_response_v2.4.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.4.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.mask$negative_threshold.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --diff $diff --step_ratio_diff $step_ratio_diff --negative_threshold $negative_threshold

#python scripts/process_inter_response_v2.4.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.4.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.mask$negative_threshold.early.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --diff $diff --step_ratio_diff $step_ratio_diff --negative_threshold $negative_threshold --early_stop


#python scripts/process_inter_response_v2.5.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.5.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.mask$negative_threshold.step$step_id_clip.ne$negative_num.s$seed.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --diff $diff --step_ratio_diff $step_ratio_diff --negative_threshold $negative_threshold --step_id_clip $step_id_clip --negative_sample_num $negative_num --seed 42

#python scripts/process_inter_response_v2.5.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.5.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.mask$negative_threshold.step$step_id_clip.ne$negative_num.s$seed.ex_full.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --diff $diff --step_ratio_diff $step_ratio_diff --negative_threshold $negative_threshold --step_id_clip $step_id_clip --negative_sample_num $negative_num --seed 42 --exclude_full

#python scripts/process_inter_response_v2.5.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.5.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.mask$negative_threshold.step$step_id_clip.ne$negative_num.s$seed.ex_full.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --diff $diff --step_ratio_diff $step_ratio_diff --negative_threshold $negative_threshold --step_id_clip $step_id_clip --negative_sample_num $negative_num --seed 42 --exclude_full


#python scripts/process_inter_response_v2.5.py \
#  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.5.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.mask$negative_threshold.step$step_id_clip.ne$negative_num.s$seed.ex_full.json" \
#  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
#  --diff $diff --step_ratio_diff $step_ratio_diff --negative_threshold $negative_threshold --step_id_clip $step_id_clip --negative_sample_num $negative_num --seed 42 --exclude_full


python scripts/process_inter_response_v2.6.py \
  --input_file "$data_dir/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.*-of-20.sample3.json" \
  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.6.rs0.2.r0.3.sample3.diff$diff.step_r_diff$step_ratio_diff.mask$negative_threshold.step$step_id_clip.ne$negative_num.s$seed.json" \
  --inter_state_file "$data_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.*-of-20.json" \
  --diff $diff --step_ratio_diff $step_ratio_diff --negative_threshold $negative_threshold --step_id_clip $step_id_clip --negative_sample_num $negative_num --seed 42

# ===================================================================================================


# Filtering by predicted rewards
#python scripts/filter_dpo_pair_by_predict_reward.py \
#  --input_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.v1.0/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.filter.$chosen_l.$reject_r.json" \
#  --chosen_l $chosen_l --reject_r $reject_r


#python scripts/filter_dpo_pair_by_predict_reward.py \
#  --input_file1 "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.v1.0.fix/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.rm_v1.0.diff$diff.decay$decay.filter.$chosen_l.$reject_r.fix.json" \
#  --chosen_l $chosen_l --reject_r $reject_r --debug_file ./debug.json
#
#
#python scripts/filter_dpo_pair_by_predict_reward.py \
#  --input_file1 "$data_dir/../logiqav2-train.react.v1.0.0shot.sample10.dpo_pair.sub_train.json" \
#  --input_file2 "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.v1.0.fix/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.rm_v1.0.diff$diff.decay$decay.filter.$chosen_l.$reject_r.w_full.fix.json" \
#  --chosen_l $chosen_l --reject_r $reject_r --debug_file ./debug-2.json


#python scripts/filter_dpo_pair_by_predict_reward.py \
#  --input_file1 "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.full.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.v1.0.fix/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.full_rm_v1.0.diff$diff.decay$decay.filter.$chosen_l.$reject_r.fix.json" \
#  --chosen_l $chosen_l --reject_r $reject_r --debug_file ./full-rm-debug.json
#
#
#python scripts/filter_dpo_pair_by_predict_reward.py \
#  --input_file1 "$data_dir/../logiqav2-train.react.v1.0.0shot.sample10.dpo_pair.sub_train.json" \
#  --input_file2 "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.full.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.v1.0.fix/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.full_rm_v1.0.diff$diff.decay$decay.filter.$chosen_l.$reject_r.w_full.fix.json" \
#  --chosen_l $chosen_l --reject_r $reject_r --debug_file ./full-rm-debug-2.json

#python scripts/filter_dpo_pair_by_predict_reward_v2.0.py \
#  --input_file1 "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.diff$diff.decay$decay.json" \
#  --reward_file "experiments/llama2.7b.chat.logiqav2.70b-distil.rm.H100.w4.v1.0/train_decay0.95.diff2.6.rewards.v1.0.fix/test-checkpoint-400/eval_predictions_rank0.json" \
#  --output_file "$data_dir/value-ver2.0/logiqav2-train.qa.react.v1.0.0shot.sample10.inter_ver2.0.rs0.2.r0.3.sample3.rm_v2.0.diff$diff.decay$decay.margin$margin.json" \
#  --margin $margin