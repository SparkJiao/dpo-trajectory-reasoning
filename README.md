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
main.py  # for PPO training.
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
|       |--- ppo/  # configurations for PPO experiments.
```

## Training

If you have less resource than we used for training, please adjust the batch size and gradient accumulation steps in the configuration file accordingly.
Also, you can choose to use DeepSpeed ZeRO-2 & 3 by simply changing the option in the configuration.

For PPO training, we used 4 H100/A100 GPUs with Tensor Parallel training. If you have fewer GPUs, or the cards are not connected via NVLink, please consider
using pure data parallel with DeepSpeed Zero-2 & 3. We have tested that DP + ZeRO-2 + Optimizer Offload on 4 H100/A100 GPUs can put 4 7B models on single
GPU at the same time.

### SFT

#### Data

The distilled data for SFT can be found in [Huggingface Hub](https://huggingface.co/datasets/chitanda/dpo-reasoning-trajectory).

#### Configs

| Dataset   | Config                                                     |    Devices     |                                            Weights & Trajectory Data                                            |  
|-----------|------------------------------------------------------------|:--------------:|:---------------------------------------------------------------------------------------------------------------:|
| ReClor    | `conf/exp/sft/reclor/llama2_7b_gpt35_dpo_sft_v2_0.yaml`    |    4 x H100    |  [Huggingface Hub](https://huggingface.co/chitanda/llama2.7b.chat.reclor.gpt35turbo1106.dpo-sft.H100.w4.v2.0)   |
| LogiQA-v2 | `conf/exp/sft/llama2_7b_llama2-70b-chat_dpo_sft_v1_0.yaml` | 4 x RTX A60000 | [Huggingface Hub](https://huggingface.co/chitanda/llama2.7b.chat.logiqav2.llama-2-70b-chat.dpo-sft.A6K.w4.v1.0) | 

### DPO

#### Data

The trajectory data used for vanilla DPO training can be found in the corresponding SFT model's Huggingface Hub.

#### Configs

| Dataset   | Config                                                            | Devices  |                                               Weights & Trajectory Data                                               |
|-----------|-------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------------------------------------------:|
| ReClor    | `conf/exp/dpo/reclor/llama2_7b_gpt351106_distil_dpo_v3_0.yaml`    | 4 x H100 |   [Huggingface Hub](https://huggingface.co/chitanda/llama2.7b.chat.reclor.gpt351106.dpo.fix_hack.H100.w4.v3.0.s42)    |
| LogiQA-v2 | `conf/exp/dpo/logiqav2/llama2_7b_70bdistil_dpo_v1_0_th_test.yaml` | 4 x H100 | [Huggingface Hub](https://huggingface.co/chitanda/llama2.7b.chat.reclor.gpt351106.step.dpo.fix_hack.H100.w4.v5.0.s42) |

### Reward Modeling

#### Data

The trajectory data as well as its completion can also be found in the corresponding SFT model's Huggingface Hub.

#### Configs

| Dataset   | Config                                                       | Devices  | Weights & Trajectory Data |
|-----------|--------------------------------------------------------------|:--------:|:-------------------------:|
| ReClor    | `conf/exp/reward/reclor/llama2_7b_gpt351106_prm_v2_0.yaml`   | 4 x H100 |    [Huggingface Hub]()    |
| LogiQA-v2 | `conf/exp/reward/logiqav2/llama2_7b_70bdistil_prm_v1_0.yaml` | 4 x H100 |    [Huggingface Hub]()    |

#### Process Rewards Annotating  (Taking LogiQA-v2 as an example)

1. Deploy the SFT model through vLLM service.
2. Run the following script to sample solutions:

```bash
python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b -cn logiqav2_qa_sft70bdistil_train_react_v1_0_0shot_sample20
```

You can change hyperparameters to control the generation process, e.g., temperature, top_k, top_p, etc.

3. Sample intermediate reasoning states by running the following script:

```bash
python scripts/sample_react_inter_states_v2.0.py \
  --input_file <the solution file generated by the SFT model> \
  --output_file <the output file> \
  --split_num <the number of splits> \
  --ratio_s <the minimum ratio of the intermediate states to start from> \
  --ratio <the ratio for sampling intermediate reasoning states>
```

4. Use the deployed SFT model to generate the trajectory data following given intermediate reasoning states by running the following script:

```bash
python service_api_caller_v1.py -cp conf/api/vllm/llama2-7b -cn logiqav2_qa_sft70bdistil_train_react_v1_0_inter_0shot_sample3
```

5. Train the reward model using the generated trajectory data through the above configs.
6. Run inference using the corresponding inference config.
7. Run the following script to construct preference pair data:

```bash
bash scripts/fixed_explore_from_infer/logiqav2/best_of_filter_full_iter1.sh
```

8. Train the DPO model using the constructed preference pair data.

### Step DPO

#### Data

The trajectory data as well as its completion can also be found in the corresponding SFT model's Huggingface Hub.

#### Configs

| Dataset   | Config                                                              | Devices  |                                                  Weights & Trajectory Data                                                  |
|-----------|---------------------------------------------------------------------|:--------:|:---------------------------------------------------------------------------------------------------------------------------:|
| ReClor    | `conf/exp/dpo/reclor/llama2_7b_gpt351106_distil_step_dpo_v5_0.yaml` | 4 x H100 |                                                     [Huggingface Hub]()                                                     |
| LogiQA-v2 | `conf/exp/dpo/logiqav2/llama2_7b_70bdistil_step_dpo_v1_0_th.yaml`   | 4 x H100 | [Huggingface Hub](https://huggingface.co/chitanda/llama2.7b.chat.logiqav2.70b-distil.step.dpo.fix_hack.A100.w4.v1.0.th.s44) |

### Step PPO

#### Data

No offline data is required.

#### Configs

| Dataset   | Config                                | Devices  | Weights & Trajectory Data |
|-----------|---------------------------------------|:--------:|:-------------------------:|
| LogiQA-v2 | `conf/exp/ppo/logiqav2_prm_v2_1.yaml` | 4 x H100 |    [Huggingface Hub]()    |

## Citation

If you find this code or the paper useful in your research, please consider citing us:

```
@article{jiao2023lpr,
  title={Learning Planning-based Reasoning with Trajectory Collection and Process Rewards Synthesizing},
  author={Fangkai Jiao and Chengwei Qin and Zhengyuan Liu and Nancy F. Chen and Shafiq Joty},
  journal      = {CoRR},
  year         = {2024},
  url          = {arxiv.org/abs/2402.00658},
}
```
