exp_name=$1
model_name=$2

for ((i=3;i<=$#;i++)); do
    step=${!i}
    echo $step
    python vllm_inference.py -cp conf/api/vllm/math -cn gsm8k_${model_name}_test_0shot_tem_v1_0 exp_name=$exp_name step=$step
done



