import datetime
import logging
import sys

import deepspeed
import hydra
import torch
import torch.distributed as dist
from deepspeed.accelerator import get_accelerator
from omegaconf import DictConfig

from megatron.core import mpu

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="test", version_base="1.2")
def main(cfg: DictConfig):
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        deepspeed.init_distributed(dist_backend="nccl", timeout=datetime.timedelta(seconds=9600))
        cfg.n_gpu = 1
        cfg.world_size = dist.get_world_size()

    cfg.device = device
    cfg.rank = dist.get_rank()

    get_accelerator().set_device(device)  # only do so when device_count > 0

    mpu.initialize_model_parallel(cfg.tensor_model_parallel_size,
                                  cfg.pipeline_model_parallel_size,
                                  cfg.ds_sequence_parallel_size,
                                  cfg.virtual_pipeline_model_parallel_size,
                                  cfg.pipeline_model_parallel_split_rank,
                                  use_distributed_optimizer=cfg.use_distributed_optimizer)
    if cfg.rank == 0:
        print(f'> initialized tensor model parallel with size '
              f'{mpu.get_tensor_model_parallel_world_size()}')
        print(f'> initialized pipeline model parallel with size '
              f'{mpu.get_pipeline_model_parallel_world_size()}')


if __name__ == "__main__":
    logger.info("Sys.argv: %s", sys.argv)
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    logger.info("Hydra formatted Sys.argv: %s", hydra_formatted_args)
    sys.argv = hydra_formatted_args

    main()
