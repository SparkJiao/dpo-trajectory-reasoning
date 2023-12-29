exp_name=$1

for ((i=2;i<=$#;i++)); do
    step=${!i}
    echo $step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/reclor_tems -cn test_react_0shot_v1_0_vllm exp_name=$exp_name step=$step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/reclor_tems -cn dev_react_0shot_v1_0_vllm exp_name=$exp_name step=$step
done
