exp_name=$1
n=$2

for ((i=3;i<=$#;i++)); do
    step=${!i}
    echo $step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/reclor_tems -cn dev_react_0shot_v1_1_vllm_sc exp_name=$exp_name sampling_params.n=$n step=$step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/reclor_tems -cn test_react_0shot_v1_1_vllm_sc exp_name=$exp_name sampling_params.n=$n step=$step
done
