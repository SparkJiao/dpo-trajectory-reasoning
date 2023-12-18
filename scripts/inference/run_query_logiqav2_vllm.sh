exp_name=$1
step=$2

python vllm_inference.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v2_0 exp_name=$exp_name step=$step

python vllm_inference.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_dev_0shot_tem_v2_0 exp_name=$exp_name step=$step

