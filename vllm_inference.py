# coding=utf-8
#
# Copyright 2023 Nanyang Technological University Fangkai Jiao
#
# Part of this code is based on the source code of Transformers
# (arXiv:1910.03771)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import inspect
import json
import logging
import os
import sys
from typing import List
import torch.distributed as dist
import datetime

import hydra
import torch
import vllm
from omegaconf import DictConfig
from tqdm import trange, tqdm
from vllm import SamplingParams, RequestOutput

from general_util.logger import setting_logger
from general_util.training_utils import load_and_cache_examples
from general_util.training_utils import set_seed

logger: logging.Logger

torch.backends.cuda.matmul.allow_tf32 = True


def evaluate(cfg: DictConfig, model: vllm.LLM, prefix="", _split="dev"):
    dataset = load_and_cache_examples(cfg, None, _split=_split)

    output_dir = getattr(cfg, "predict_dir", cfg.output_dir)

    if cfg.local_rank in [-1, 0] and not os.path.exists(os.path.join(output_dir, prefix)):
        os.makedirs(os.path.join(output_dir, prefix))

    post_processor = hydra.utils.instantiate(cfg.post_process) if "post_process" in cfg and cfg.post_process else None

    # Eval!
    torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))

    all_prompts = []
    all_meta_data = []

    for i in trange(len(dataset)):
        if cfg.local_rank != -1 and i % cfg.world_size != cfg.local_rank:
            continue
        inputs = dataset.api_getitem(i)
        all_prompts.append(inputs.pop("text"))
        all_meta_data.append(inputs.pop("meta_data"))

    sampling_params: SamplingParams = hydra.utils.instantiate(cfg.sampling_params)

    outputs: List[RequestOutput] = model.generate(all_prompts, sampling_params)
    if len(outputs) != len(all_meta_data):
        logger.warning(f"outputs: {len(outputs)}, meta_data: {len(all_meta_data)}")

    for output, meta_data in tqdm(zip(outputs, all_meta_data), total=len(all_meta_data), desc="Post-processing"):
        output = {"response": output}
        if any(hasattr(post_processor, tmp) for tmp in ["gather", "gather_object"]):
            kwargs = {
                "ddp": cfg.ddp_eval and cfg.local_rank != -1
            }
        else:
            kwargs = {}
        post_processor(meta_data, output, **kwargs)

    results = {}

    sig = inspect.signature(post_processor.get_results)
    post_kwargs = {}
    if "output_dir" in list(sig.parameters.keys()):
        post_kwargs["output_dir"] = os.path.join(output_dir, prefix)

    post_results, post_predictions = post_processor.get_results(**post_kwargs)
    results.update(post_results)
    metric_log = '\t'.join([f"{k}: {v}" for k, v in results.items()])
    predictions = post_predictions

    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    logger.info(metric_log)

    if len(predictions) > 0:
        if cfg.local_rank == -1:
            prediction_file = os.path.join(output_dir, prefix, "eval_predictions.json")
        else:
            prediction_file = os.path.join(output_dir, prefix, f"eval_predictions_rank{cfg.local_rank}.json")
        json.dump(predictions, open(prediction_file, "w"), indent=2)

    torch.cuda.empty_cache()

    return results


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] not in [-1, "-1"]:
    #     cfg.local_rank = int(os.environ["LOCAL_RANK"])
    #
    # if cfg.local_rank == -1 or cfg.no_cuda:
    device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
    cfg.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    #     torch.cuda.set_device(cfg.local_rank)
    #     device = str(torch.device("cuda", cfg.local_rank))
    #     dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=7200))
    #     cfg.n_gpu = 1
    #     cfg.world_size = dist.get_world_size()
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.local_rank)
    cfg.device = device

    global logger
    logger = setting_logger(cfg.output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, cfg.device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)
    logger.warning(f"CPU cores: {os.cpu_count()}")

    # Set seed
    set_seed(cfg)

    # Test
    results = {}

    checkpoints = [cfg.output_dir]
    if cfg.save_best:
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    # elif cfg.prediction_cfg.best_checkpoint and os.path.exists(cfg.prediction_cfg.best_checkpoint):
    #     checkpoints = [cfg.prediction_cfg.best_checkpoint]
    #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    elif cfg.eval_sub_path:
        checkpoints = list(sorted(list(set(
            os.path.dirname(c) for c in
            glob.glob(cfg.output_dir + f"/{cfg.eval_sub_path}/" + "pytorch_model*.bin", recursive=True)
        ))))
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info(" the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        split = "dev"

        model = vllm.LLM(model=checkpoint,
                         tensor_parallel_size=cfg.n_gpu,
                         swap_space=getattr(cfg, "swap_space", 4),
                         gpu_memory_utilization=getattr(cfg, "gpu_memory_utilization", 0.9),
                         load_format=getattr(cfg, "load_format", "auto"),)

        if cfg.test_file:
            prefix = f'test' + (f'-{prefix}' if prefix != "" else "")
            split = "test"

        result = evaluate(cfg, model, prefix=prefix, _split=split)
        result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
        results.update(result)

        del model

    return results


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["WANDB__SERVICE_WAIT"] = "1200"
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args
    print(sys.argv)
    main()
