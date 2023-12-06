#model=$1
#port=$2
#name=$3
#
#python -m vllm.entrypoints.openai.api_server --model experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600 --tokenizer experiments/llama2.7b.chat.logiqav2.gpt35turbo-dpo-sft.H100.w2.v2.0/checkpoint-1600 --dtype bfloat16 --served-model-name llama-2-7b-sft-v4.1-cp1600 --port 6000 --disable-log-requests
#
#python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b -cn logiqav2_qa_train_react_sft_v2_0_inter_0shot_sample3 port=6000 dataset_split_id=1

# Completed:
# ssvr: 12,
# xin: 14, 13, 15
# 4ir: 0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,19

# ======================================

#python -m vllm.entrypoints.openai.api_server --model experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600 --tokenizer experiments/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0/checkpoint-1600 --dtype bfloat16 --served-model-name llama-2-7b-sft70b-v1.0-cp1600 --port 6000 --disable-log-requests
#
#
#python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b -cn logiqav2_qa_sft70bdistil_train_react_v1_0_inter_0shot_sample3 port= dataset_split_id=

# node15: 0,2,3,4,8,9，15，16,17,18
# node13: 1,
# node08: 2,
# node02: 1,5,14
# node09: 6,10,11,12,13,19
# ssvr: 7,17


#===================================================================================================
model_path=experiments/llama2.7b.chat.logiqav2.70b-distil.step.dpo.H100.w4.v2.3
step=1600

python -m vllm.entrypoints.openai.api_server --model $model_path/checkpoint-$step \
  --tokenizer $model_path/checkpoint-$step \
  --dtype bfloat16 --served-model-name llama-2-7b-70bdistil-step-dpo-v2.3-cp$step --disable-log-requests -tp 2

