exp_name=$1
#step=$2

for ((i=2;i<=$#;i++)); do
    step=${!i}
    echo $step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/ar_lsat_tems -cn dev_react_v1_0 exp_name=$exp_name step=$step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/ar_lsat_tems -cn test_react_v1_0 exp_name=$exp_name step=$step
done
#python vllm_inference.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v2_0 exp_name=$exp_name step=$step



