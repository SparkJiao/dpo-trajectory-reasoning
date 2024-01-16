exp_name=$1

for ((i=2;i<=$#;i++)); do
    step=${!i}
    echo $step
    python vllm_inference.py -cp conf/api/vllm/llama2-7b/folio_tems -cn react_dev_0shot_tem_v1_0 exp_name=$exp_name step=$step
done



