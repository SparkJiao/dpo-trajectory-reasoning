exp_name=$1
n=$2

for ((i=3;i<=$#;i++)); do
    step=${!i}
    echo $step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_dev_0shot_tem_v2_1_sc exp_name=$exp_name sampling_params.n=$n step=$step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v2_1_sc exp_name=$exp_name sampling_params.n=$n step=$step
done
#python vllm_inference.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v2_0 exp_name=$exp_name step=$step



