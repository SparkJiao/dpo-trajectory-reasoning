import time
from typing import Union

import deepspeed
import torch
import torch.distributed as dist
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from fairscale.nn.model_parallel import initialize as mpu
from transformers import PreTrainedModel, PreTrainedTokenizer

from general_util.logger import get_child_logger

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


def gather_log_probs(logits, labels, pad_token_id: int):
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    loss_mask = labels.ne(pad_token_id)
    return log_probs_labels.squeeze(-1) * loss_mask


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(), model_ema.parameters()):
            # TODO: use pre-filtering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def trainer_save_single_model(model: Union[deepspeed.DeepSpeedEngine, deepspeed.PipelineEngine],
                              output_dir: str,
                              local_rank: int,
                              zero_stage: int = 1,
                              save_ds_state: bool = False,
                              tokenizer: PreTrainedTokenizer = None,
                              state_save_dir: str = None, ):
    unwrapped_model = model.module
    assert isinstance(unwrapped_model, PreTrainedModel)

    if save_ds_state:
        model.save_checkpoint(state_save_dir)

    if zero_stage == 3:
        state_dict = model._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = unwrapped_model.state_dict()

    if mpu.model_parallel_is_initialized():
        dp_rank = mpu.get_data_parallel_rank()
    else:
        dp_rank = local_rank

    if dist.is_initialized() and dp_rank != 0:
        dist.barrier()

    if dp_rank in [-1, 0]:
        unwrapped_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)

        if local_rank in [-1, 0]:
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            logger.info("Saving model checkpoint to %s", output_dir)

        if dist.is_initialized():
            dist.barrier()
