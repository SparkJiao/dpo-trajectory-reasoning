#!/bin/bash

#SBATCH -n 1 # 指定核心数量
#SBATCH -N 1 # 指定node的数量
#SBATCH -w node15
#SBATCH -p NH100q # 提交到哪一个分区
#SBATCH -o llama2_7b_gpt351106_prm_v1_4.o # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e llama2_7b_gpt351106_prm_v1_4.e # 把报错结果STDERR保存在哪一个文件
#SBATCH --gres=gpu:8 # 需要使用多少GPU，n是需要的数量

#trainers/training_torch_fsdp_v3 75 144 0,1
#deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mul.py -cp conf/exp/reward/reclor -cn llama2_7b_gpt351106_prm_v1_4
#deepspeed --include localhost:0,1,2,3,4,5,6,7 trainer_base_ds_mul.py seed=42 -cp conf/exp/dpo/reclor/ -cn llama2_7b_gpt351106_distil_step_dpo_v2_1

#CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model rl-hybrid-engine/experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.A100.w2.v1.0/checkpoint-2400/ \
#  --tokenizer rl-hybrid-engine/experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.A100.w2.v1.0/checkpoint-2400/ \
#  --served-model-name llama2-7b-reclor-distil --port 6000 -tp 2 --disable-log-requests --gpu-memory-utilization 0.95

#python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b/reclor_tems -cn train_react_0shot_v1_0_sample_service num_workers=128

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server --model rl-hybrid-engine/experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.H100.w4.v2.0/checkpoint-400 --tokenizer rl-hybrid-engine/experiments/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.H100.w4.v2.0/checkpoint-1200 --served-model-name llama2-7b-reclor-distil --port 6000 -tp 4 --disable-log-requests --gpu-memory-utilization 0.95


#bash scripts/inference/run_query_reclor_vllm_sc_v1.1.sh llama2.7b.chat.reclor.gpt351106.dpo.fix_hack.H100.w4.v3.0.s42 5 200 400 600 800 1000 1200 1400
#
#bash scripts/inference/run_query_reclor_vllm_sc_v1.1.sh llama2.7b.chat.reclor.gpt351106.dpo.fix_hack.H100.w4.v3.0.s43 5 200 400 600 800 1000 1200 1400
#
#bash scripts/inference/run_query_reclor_vllm_sc_v1.1.sh llama2.7b.chat.reclor.gpt351106.dpo.fix_hack.H100.w4.v3.0.s44 5 200 400 600 800 1000 1200 1400
#
#
#bash scripts/inference/run_query_logiqav2_vllm_sc_v2.1.sh llama2.7b.chat.logiqav2.70b-distil.dpo.fix_hack.A100.w2.v1.0.th.test.s44 5 1600 2000 2400 2800 3200


#bash scripts/inference/run_query_reclor_vllm_sc_v1.1.sh llama2.7b.chat.logiqav2.70b-distil.dpo.fix_hack.A100.w2.v1.0.th.test.s44 5 2800 3200


bash scripts/inference/run_query_reclor_vllm_sc_v1.1.sh llama2.7b.chat.reclor.gpt351106.step.dpo.fix_hack.H100.w4.v5.0.s43 5 200 400 600 800 1000 1200 1400 1600
bash scripts/inference/run_query_reclor_vllm_sc_v1.1.sh llama2.7b.chat.reclor.gpt351106.step.dpo.fix_hack.H100.w4.v5.0.s44 5 200 400 600 800 1000 1200 1400 1600