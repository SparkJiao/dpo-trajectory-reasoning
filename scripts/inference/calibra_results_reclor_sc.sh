exp_name=$1

for ((i=2;i<=$#;i++)); do
    step=${!i}
    echo "*********************  $step *********************"
    echo "============= Dev ============="
    cat /export/home2/fangkai/rl-hybrid-engine/experiments/${exp_name}/checkpoint-${step}/reclor.react.dev.n5.tem1.0.0shot.v1.1.metrics.json
#    echo "============= Test ============="
#    python scripts/calculate_react_acc_w_clean.py --input_file /export/home2/fangkai/rl-hybrid-engine/experiments/${exp_name}/checkpoint-${step}/logiqav2.test.react.n5.tem1.0.v1.0.0shot.json
done
#python vllm_inference.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v2_0 exp_name=$exp_name step=$step



