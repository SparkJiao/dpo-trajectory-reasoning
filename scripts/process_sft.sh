data_dir=experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600

#python scripts/process_react_nodes.py --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.json \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.json
#
#python scripts/sent_tf_react_step_encoding.py --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.json \
#  --model_path ../pretrained-models/bge-large-en-v1.5 \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.emb.npy

#python scripts/react_step_union_find.py --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.json \
#  --embedding_path experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.emb.npy --threshold 0.95 \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.json
#
#python scripts/construct_dpo_data_via_step_value_v1.py \
#  --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.json \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len2.in4.v0.1.json \
#  --save_full_data

#python scripts/split_train_dev.py --input_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.cluster.t0.95.TO.len2.in4.v0.1.json \
#  --dev_num 5000

# =========================================================================================================================================================

#python scripts/construct_dpo_data_via_step_value_v1.py \
#  --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.json \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len1.in4.v0.3.json \
#  --save_full_data --value_diff 0.3 --step_lens_diff 1

# Namespace(input_file='experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.json', output_file='experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len1.in4.v0.3.json', step_lens_diff=1, max_inter_samples=4, value_diff=0.3, included_types=('Thought', 'Observation'), save_full_data=True)
# Average step sample number: 12.84559585492228
# Average sample numer: 22.644320446392985
# 284073
# 122925
# Save full data to experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len1.in4.v0.3.full_only.json
# Full data number: 122925
# =========================================================================================================================
#
#python scripts/construct_dpo_data_via_step_value_v1.py \
#  --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.json \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len1.in4.v0.5.json \
#  --save_full_data --value_diff 0.5 --step_lens_diff 1

#Namespace(input_file='experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.json', output_file='experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len1.in4.v0.5.json', step_lens_diff=1, max_inter_samples=4, value_diff=0.5, included_types=('Thought', 'Observation'), save_full_data=True)
#Average step sample number: 9.495735352730172
#Average sample numer: 19.294459944200877
#242049
#122925
#Save full data to experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len1.in4.v0.5.full_only.json
#Full data number: 122925


# =========================================================================================================================


#python scripts/construct_dpo_data_via_step_value_v1.py \
#  --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.json \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t0.95.TO.len1.in4.v0.5.json \
#  --value_diff 1.0 --step_lens_diff 1

# =========================================================================================================================


# FIX the bug in construct_dpo_data_via_step_value_v1.py
# See FIXME note.

dis_threshold=0.95
value_diff=0.5
step_lens_diff=1
cluster_file=$data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t$dis_threshold.TO.json
pair_file=$data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.t$dis_threshold.TO.len$step_lens_diff.in4.v$value_diff.fix-ver1.0.json

python scripts/construct_dpo_data_via_step_value_v1.py \
  --input_file $cluster_file \
  --output_file $pair_file \
  --value_diff $value_diff --step_lens_diff $step_lens_diff

python scripts/split_train_dev.py --input_file $pair_file --dev_num 5000
