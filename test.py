import os

import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="test", version_base="1.2")
def main(cfg: DictConfig):
    print(os.getcwd())
    # t = open("step3_rlhf_finetuning/README.md", "r").read()
    # print(t)
    print(cfg)
    # print(cfg.test_resolve)


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
