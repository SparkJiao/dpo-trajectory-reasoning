# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
import copy
# Nanyang Technological University, Singapore (NTU, Singapore)
# Fangkai Jiao

from typing import Dict, List, Optional, Tuple, Union

import time
import torch.distributed as dist
import deepspeed
import torch
from omegaconf import OmegaConf, DictConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.ops.adam import FusedAdam
from transformers import AutoModelForCausalLM, get_scheduler
from models.utils import initialize_peft_model
from general_util import training_utils, dist_utils
from general_util.logger import get_child_logger

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from peft import PeftModel
import hydra

logger = get_child_logger(__name__)


def log_init(model_name, stime=None):
    if not dist.is_initialized() or dist.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class DeepSpeedRLLoraEngine:
    def __init__(self,
                 cfg: DictConfig,
                 base_model: PreTrainedModel,
                 # actor_model: PeftModel,
                 # critic_model: PeftModel,
                 tokenizer: Union[str, PreTrainedTokenizer], ):
        self.cfg = cfg
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.pretrained_model = base_model

        actor_model = hydra.utils.instantiate(cfg.actor_model_init, base_model)

        self.actor, self.actor_optim, self.actor_lr_scheduler = self._init_actor(actor_model)
        self.ref = base_model
        self.actor_ema = None
        if self.cfg.enable_ema:
            self.actor_ema = hydra.utils.instantiate(cfg.actor_model_init, base_model)

        self.critic = None
        self.reward = hydra.utils.instantiate(cfg.critic_model_init, base_model)

    def _init_actor(self, actor_model: PeftModel, ):
        stime = log_init("Actor")

        ds_config = self.cfg.actor_ds_config
        if "total_num_steps" in ds_config.scheduler.params:
            ds_config.scheduler.params.total_num_steps = self.cfg.max_steps
        ds_config.scheduler.params.warmup_num_steps = self.cfg.warmup_steps
        ds_config = OmegaConf.to_container(ds_config, resolve=True)
        ds_config["train_mirco_batch_size_per_gpu"] = self.cfg.per_gpu_train_batch_size
        # ds_config["train_batch_size"] = self.cfg.per_gpu_train_batch_size * self.cfg.gradient_accumulation_steps_actor

        optim_params = training_utils.get_optimizer_grouped_parameters(actor_model, self.cfg.actor_weight_decay)

        actor_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=actor_model,
            model_parameters=optim_params,
            config_params=ds_config,
        )

        log_init("Actor", stime)

        return actor_engine, optimizer, scheduler

    def _init_ref(self, actor_model_or_base_model: PreTrainedModel):
        pass

    def _init_ema(self, actor_model: PeftModel):
        # Currently not implemented since I'm not familiar with the theory.
        pass
