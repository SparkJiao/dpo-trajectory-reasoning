data_dir=experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600

#python scripts/process_react_nodes.py --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.json \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.json
#
#python scripts/sent_tf_react_step_encoding.py --input_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.json \
#  --model_path ../pretrained-models/bge-large-en-v1.5 \
#  --output_file experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.emb.npy

dis_threshold=0.95
#value_diff=0.5
value_diff=0.2
step_lens_diff=1
cluster_file=$data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.ver2.0.t$dis_threshold.TO.json
pair_file=$data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.cluster.ver2.0.t$dis_threshold.TO.len$step_lens_diff.in4.v$value_diff.json

#python scripts/react_step_union_find_v2.py --input_file $data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.json \
#  --embedding_path $data_dir/logiqav2-train.full.qa.react.v1.0.1shot.sample20.clean_nodes.emb.npy --threshold $dis_threshold \
#  --output_file $cluster_file

python scripts/construct_dpo_data_via_step_value_v2.0.py --input_file $cluster_file --output_file $pair_file --value_diff $value_diff --step_lens_diff $step_lens_diff

python scripts/split_train_dev.py --input_file $pair_file --dev_num 5000

# `value_diff=0.5`:
#Average step sample number: 2.6797130330809087
#Average sample numer: 12.478437624551614
#156542
#122925
# -------------------------------------------------------

# `value_diff=0.2`:
#Average step sample number: 7.198326026305301
#Average sample numer: 16.997050617776008
#213228
#122925
