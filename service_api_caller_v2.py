# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
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

import inspect
import logging
import sys
import os

import hydra
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm
from multiprocessing import Pool

from general_util.logger import setting_logger
from general_util.training_utils import set_seed, load_and_cache_examples

logger: logging.Logger
request_dataset: torch.utils.data.Dataset


def _client_init(dataset):
    global request_dataset
    request_dataset = dataset


def _get_response(idx):
    response = request_dataset[idx]
    return response


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    global logger
    logger = setting_logger(cfg.output_file, local_rank=cfg.local_rank)

    # Set seed
    set_seed(cfg)

    dataset = load_and_cache_examples(cfg, None, _split="test")
    post_processor = hydra.utils.instantiate(cfg.post_process)

    ids = list(range(len(dataset)))
    with Pool(cfg.num_workers, initializer=_client_init, initargs=(dataset,)) as p:
        responses = list(tqdm(
            p.imap(_get_response, ids, chunksize=32),
            total=len(dataset),
            desc="Sending requests"
        ))

    for res in tqdm(responses, total=len(responses), desc="Processing responses"):
        meta_data = res.pop("meta_data")
        post_processor(meta_data, res)

    sig = inspect.signature(post_processor.get_results)
    post_kwargs = {}
    if "output_dir" in list(sig.parameters.keys()):
        post_kwargs["output_dir"] = cfg.output_dir

    results, predictions = post_processor.get_results(**post_kwargs)
    logger.info(f"=================== Results =====================")
    for key, value in results.items():
        logger.info(f"{key}: {value}")

    return results


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"

    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args

    main()
