sft_model_dir=experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600/fix_hack_data_dir/
dpo_data="logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_dpo_pair.json"
step_dpo_data="logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_10.neg10.pos0.5.v2.2.(2,3).pair.product.(2,3).full_only.json"

seed=43

python scripts/split_pairs_according_to_ids.py --input_file $sft_model_dir/$dpo_data --output_file $sft_model_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_dpo_pair.ratio40.json --ratio 0.4
python scripts/split_pairs_according_to_ids.py --input_file $sft_model_dir/$dpo_data --output_file $sft_model_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_dpo_pair.ratio60.s$seed.json --ratio 0.6
python scripts/split_pairs_according_to_ids.py --input_file $sft_model_dir/$dpo_data --output_file $sft_model_dir/logiqav2-train.full.qa.react.v1.0.0shot.sample10.clean_dpo_pair.ratio80.json --ratio 0.8

python scripts/split_pairs_according_to_ids.py --input_file $sft_model_dir/$step_dpo_data --output_file "$sft_model_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_10.neg10.pos0.5.v2.2.(2,3).pair.product.(2,3).full_only.ratio40.json" --ratio 0.4
python scripts/split_pairs_according_to_ids.py --input_file $sft_model_dir/$step_dpo_data --output_file "$sft_model_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_10.neg10.pos0.5.v2.2.(2,3).pair.product.(2,3).full_only.ratio60.s$seed.json" --ratio 0.6
python scripts/split_pairs_according_to_ids.py --input_file $sft_model_dir/$step_dpo_data --output_file "$sft_model_dir/logiqav2-train.react.v1.0.0shot.sample10.clean_inter_ver2.0.rs0.2.r0.3.prm_hack_fix_v10_cp800_best_of_10.neg10.pos0.5.v2.2.(2,3).pair.product.(2,3).full_only.ratio80.json" --ratio 0.8