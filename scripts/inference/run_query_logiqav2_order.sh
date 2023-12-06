exp_dir=$1
name=$2
port=$3
step=$4

echo "python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b/logiqav2_temps -cn react_test_0shot_tem_v1_0 exp_dir=$exp_dir model=$name port=$port step=$step"
python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v1_0 exp_dir=$exp_dir model=$name port=$port step=$step

echo "python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b/logiqav2_temps -cn react_test_0shot_tem_v1_0_o1 exp_dir=$exp_dir model=$name port=$port step=$step"
python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v1_0_o1 exp_dir=$exp_dir model=$name port=$port step=$step

echo "python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b/logiqav2_temps -cn react_test_0shot_tem_v1_0_o2 exp_dir=$exp_dir model=$name port=$port step=$step"
python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b/logiqav2_tems -cn react_test_0shot_tem_v1_0_o2 exp_dir=$exp_dir model=$name port=$port step=$step
