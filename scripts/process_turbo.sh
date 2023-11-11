

#python scripts/process_react_nodes.py --input_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.json \
#  --output_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.json

#python scripts/sent_tf_react_step_encoding.py --input_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.json \
#  --model_path ../pretrained-models/bge-large-en-v1.5 \
#  --output_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.emb.npy

python scripts/react_step_union_find.py --input_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.json \
  --embedding_path data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.emb.npy --threshold 0.95 \
  --output_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.cluster.t0.95.TO.json

python scripts/construct_dpo_data_via_step_value_v1.py \
  --input_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.cluster.t0.95.TO.json \
  --output_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.cluster.t0.95.TO.len2.in4.v0.1.json \
  --save_full_data

python scripts/split_train_dev.py --input_file data/trajectory/react/logiqav2-train-v1.0.react.1shot.turbo.sample5.clean_nodes.cluster.t0.95.TO.len2.in4.v0.1.json \
  --dev_num 5000