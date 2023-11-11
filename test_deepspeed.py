from models.llama import LlamaForCausalLMDPO
from transformers.models.llama import LlamaConfig
from omegaconf import DictConfig, OmegaConf
import omegaconf
import datetime

config = LlamaConfig(vocab_size=10, num_hidden_layers=1)

model = LlamaForCausalLMDPO(config)

print(model.__class__.__name__)

import deepspeed

ds_config = OmegaConf.load("conf/deepspeed/train_hybrid_engine_zero1.yaml")
ds_config.train_micro_batch_size_per_gpu = 1
ds_config.gradient_accumulation_steps = 1
ds_config.scheduler.params.total_num_steps = 1000
ds_config.scheduler.params.warmup_num_steps = 10
ds_config = OmegaConf.to_container(ds_config, resolve=True)

deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=9600))
engine = deepspeed.initialize(model=model,
                              config=ds_config)

print(engine.__class__.__name__)
print(engine.module.__clas__.__name__)
