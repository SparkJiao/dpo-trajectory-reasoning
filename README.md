# Learning Planning-based Reasoning with Trajectory Collection and Process Rewards Synthesizing

This repository contains the code for the paper "Learning Planning-based Reasoning with Trajectory Collection and Process Rewards Synthesizing".

## Requirements

The required packages can be found in `requirements.txt`.

**CUDA Version**: 11.7/11.8/12.1

## Project Structure

This project relies on Hydra to manage configurations. The configuration files are located in `conf/`. There are several entrypoints for different usage:
```bash
service_api_caller_v1.py  # for vllm-service based inference.
openai_api_caller_v1.py  # for OpenAI API calling.
vllm_inference.py  # for vllm normal inference.
trainer_base_ds_mul.py  # for training and non-generative inference.
```

**Typical usage**:
```bash
python <entrypoint> -cp <config_path> -cn <config_file_name>
```
For example, for training Llama2-7B SFT on LogiQA-v2, run the following command (Note that our training uses deepspeed):
```bash
deepspeed trainer_base_ds_mul.py -cp conf/exp/sft/ -cn llama2_7b_llama2-70b-chat_dpo_sft_v1_0
```

### Configuration Usage
```
|--- conf/
|    |--- api/  # configurations for API calling, including vllm-service and OpenAI API.
|       |--- vllm_params  # configurations for vllm sampling params.
|    |--- deepspeed # configurations for deepspeed.
|    |--- exp/  # experiment configurations.
|       |--- sft/  # configurations for SFT experiments.
|       |--- reward/ # configurations for reward models.
|       |--- dpo/  # configurations for DPO experiments.
```


## Citation

If you find this code or the paper useful in your research, please consider citing us:

```
@article{jiao2023lpr,
  title={Learning Planning-based Reasoning with Trajectory Collection and Process Rewards Synthesizing},
  author={Fangkai Jiao and Chengwei Qin and Zhengyuan Liu and Nancy F. Chen and Shafiq Joty},
  journal      = {CoRR},
  year         = {2024},
  url          = {},
}
```
