import gc
import json
import os
from collections import OrderedDict
from glob import glob
from typing import Union, Optional, Callable

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from einops import rearrange
from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from torch import nn
from tqdm import tqdm
from transformers.models.llama import modeling_llama
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaModel,
    is_flash_attn_greater_or_equal_2_10,
    LlamaForCausalLM as HfLlamaForCausalLM,
)

from general_util.dist_utils import get_pipeline_parallel_rank, get_pipeline_parallel_world_size
from general_util.logger import get_child_logger
from models.llama import PreTrainedModelPeftMixin

logger = get_child_logger(__name__)


class LlamaAttentionParallel(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False,
            init_method=lambda x: x
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self._init_rope()

        # self.output_size_per_partition = self.q_proj.output_size_per_partition
        self.num_heads = self.num_heads // mpu.get_model_parallel_world_size()
        self.num_key_value_heads = self.num_key_value_heads // mpu.get_model_parallel_world_size()
        self.hidden_size = self.hidden_size // mpu.get_model_parallel_world_size()


class LlamaFlashAttention2Parallel(LlamaFlashAttention2):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            gather_output=False,
            init_method=lambda x: x
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.o_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )
        self._init_rope()

        # self.output_size_per_partition = self.q_proj.output_size_per_partition
        self.num_heads = self.num_heads // mpu.get_model_parallel_world_size()
        self.num_key_value_heads = self.num_key_value_heads // mpu.get_model_parallel_world_size()
        self.hidden_size = self.hidden_size // mpu.get_model_parallel_world_size()

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


class LlamaMLPParallel(LlamaMLP):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )


modeling_llama.LlamaAttention = LlamaAttentionParallel
modeling_llama.LlamaMLP = LlamaMLPParallel
modeling_llama.LLAMA_ATTENTION_CLASSES["eager"] = LlamaAttentionParallel
modeling_llama.LLAMA_ATTENTION_CLASSES["flash_attention_2"] = LlamaFlashAttention2Parallel


class LlamaModelParallel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # register a causal mask to separate causal and padding mask creation. Merging happends in the attention class
        causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
        self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()


class LlamaModelParallelPretrainedMixin(PreTrainedModelPeftMixin):
    @staticmethod
    def load_parallel_state_dict(path: str,
                                 process_exclusion: bool = False,
                                 protocol: str = 'file',
                                 **kwargs):
        """
        Load state_dict from ``path``.

        The format of pretrained model should be the same as that of
        `huggingface`.

        :return: state_dict. Note that the state_dict should be processed
            properly to match the current rank.
        """
        config = LlamaConfig.from_pretrained(path)

        dp_rank = mpu.get_data_parallel_rank()
        tp_rank = mpu.get_model_parallel_rank()
        tp_size = mpu.get_model_parallel_world_size()

        pp_rank = get_pipeline_parallel_rank()
        pp_size = get_pipeline_parallel_world_size()
        pp_group = mpu.get_pipeline_parallel_group()

        state_dict = OrderedDict()
        weights = []
        parts = None
        # 如果开启了进程互斥，那么每个进程都会显示进度条，否则只显示 RANK0 的
        hide_progress = not process_exclusion and int(os.environ.get("RANK", "0")) != 0
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 dist.get_world_size() 次循环
            rank_order = range(dist.get_world_size())
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        for rank in rank_order:
            # 如果开启了进程互斥，那么只有对应 RANK 的能进入循环；不开启进程互斥的话就都可以进
            if int(os.environ.get("RANK", "0")) == rank or not process_exclusion:
                # PP 分层的方法保存在了 os.environ["COLLIE_PP_PARTS"], 格式类似于 [0, 17, 35], 左闭右开
                if pp_size > 1:
                    # 保存的是 json 格式
                    # parts = env.pipeline_parts
                    pass
                # 如果存在 pytorch_model.bin.index.json 文件的话，此时不同的 pp 进程可以按需加载自己需要的权重
                if os.path.exists(os.path.join(path, "pytorch_model.bin.index.json")):
                    # and "COLLIE_PP_PARTS" in os.environ.keys()):
                    weight_map = json.load(open(os.path.join(path, "pytorch_model.bin.index.json"), "r"))["weight_map"]
                    # layers 表示自己需要的层
                    # layers = list(range(parts[int(os.environ["COLLIE_PP_RANK"])], parts[int(os.environ["COLLIE_PP_RANK"]) + 1]))
                    # 筛选出形似 model.layers.0 这样的层。包含两个条件：1. 有数字的层；2. 数字加一要在 layers 里面（因为最开始还有个 embedding 占一层）
                    weights.extend([value for key, value in weight_map.items() if len(key.split(".")) > 2 and key.split(".")[2].isdigit()
                                    # and (int(key.split(".")[2]) + 1) in layers
                                    ])
                    # 去重
                    weights = list(set(weights))
                    # 继续筛选，如果有 0 层，那么就要加载 embedding；如果有最后一层，那么就要加载 lm_head；如果有倒数第二层，那么就要加载 norm
                    # if 0 in layers:
                    weights.append(weight_map["model.embed_tokens.weight"])
                    # if max(parts) - 1 in layers:
                    weights.append(weight_map["lm_head.weight"])
                    # if max(parts) - 2 in layers:
                    weights.append(weight_map["model.norm.weight"])
                else:
                    # 如果没有 pytorch_model.bin.index.json 文件的话，那么就加载所有的权重
                    # weights = [weight for weight in os.list(path) if weight.endswith(".bin")]
                    weights = list(glob(os.path.join(path, "pytorch_model*.bin")))
                # with progress(weights, desc="Loading state dict", total=len(weights), disable=hide_progress) as pbar:
                #     for weight in pbar:
                for weight in tqdm(weights, desc="Loading state dict", disable=hide_progress, total=len(weights)):
                    part_state_dict = torch.load(os.path.join(path, weight))
                    # for key in list(part_state_dict.keys()):
                    #     # 对 q_proj.weight 和 k_proj.weight 进行 reshape
                    #     if key.endswith("q_proj.weight") or key.endswith("k_proj.weight"):
                    #         part_state_dict[key] = rearrange(
                    #             part_state_dict[key],
                    #             "(h two t) d -> h two t d",
                    #             h=config.num_attention_heads,
                    #             two=2).transpose(1, 2).reshape(
                    #             config.hidden_size,
                    #             config.hidden_size)
                    # part_state_dict[key.replace("model.", "")] = part_state_dict.pop(key)
                    state_dict.update(part_state_dict)
                    del part_state_dict

                # 根据用户配置的新的 tp size 进行分割
                for key in list(state_dict.keys()):
                    filte_list = ["q_proj.weight", "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight",
                                  "embed_tokens.weight", "lm_head.weight"]
                    need_split = any([key.endswith(filte) for filte in filte_list])
                    if pp_size > 1:
                        # embedding 层和 lm_head 都需要切
                        need_split = need_split or int(key.split(".")[0]) == max(parts) - 1
                        need_split = need_split or int(key.split(".")[0]) == min(parts)
                    if need_split:
                        tensor = list(torch.chunk(state_dict[key], tp_size, dim=0))[int(tp_rank)].detach().clone()
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
                    elif key.endswith("o_proj.weight") or key.endswith("down_proj.weight"):
                        tensor = list(torch.chunk(state_dict[key], tp_size, dim=1))[int(tp_rank)].detach().clone()
                        del state_dict[key]
                        if process_exclusion:
                            # CPU 内存回收（速度很慢）
                            gc.collect()
                        state_dict[key] = tensor
            if dist.is_initialized() and process_exclusion:
                # 如果选择了进程互斥，那么本次循环中不需要加载权重的进程需等待
                dist.barrier()
        return state_dict

    @staticmethod
    def save_parallel_state_dict(
            state_dict: dict,
            # path: str,
            config: LlamaConfig,
            process_exclusion: bool = False,
    ):
        """
        Save state_dict to ``path``.

        The format of saved state dict should be the same as that of
        `huggingface`.
        """

        # io_driver = IODriver.from_protocol(protocol)

        def reshape_wq_wk(w: torch.Tensor, kv=False):
            if hasattr(config, "num_key_value_heads"):
                # llama2 (transformers >= 4.31.0)
                num_key_value_heads = config.num_key_value_heads
            else:
                num_key_value_heads = config.num_attention_heads
            head_dim = config.hidden_size // config.num_attention_heads
            if kv:
                num_heads = num_key_value_heads
            else:
                num_heads = config.num_attention_heads
            return (
                w.view(num_heads, head_dim // 2, 2, config.hidden_size)
                .transpose(1, 2)
                .reshape(num_heads * head_dim, config.hidden_size)
            )

        dp_rank = mpu.get_data_parallel_rank()
        tp_rank = mpu.get_model_parallel_rank()
        tp_size = mpu.get_model_parallel_world_size()

        pp_rank = get_pipeline_parallel_rank()
        pp_size = get_pipeline_parallel_world_size()
        pp_group = mpu.get_pipeline_parallel_group()

        # gather to tp rank 0
        if dist.is_initialized() and process_exclusion:
            # 如果启动了进程互斥，则要进行 pp_size 次循环
            rank_order = range(pp_size)
        else:
            # 不开启只进行一次循环
            rank_order = range(1)
        dst = mpu.get_model_parallel_src_rank()
        for rank in tqdm(rank_order, desc="Saving model", disable=int(os.environ.get("RANK", "0")) != 0):
            if dp_rank == 0 and (pp_rank == rank or not process_exclusion):
                for key in sorted(list(state_dict.keys())):
                    tensor_list = None
                    if tp_rank == 0:
                        tensor_list = [
                            torch.zeros_like(state_dict[key])
                            .to(state_dict[key].dtype)
                            .cuda()
                            for _ in range(tp_size)
                        ]
                    dist.gather(
                        state_dict[key].cuda(),
                        dst=dst,
                        gather_list=tensor_list,
                        group=mpu.get_model_parallel_group(),
                    )
                    if tp_rank == 0:
                        col_filter = [
                            "q_proj.weight",
                            "k_proj.weight",
                            "v_proj.weight",
                            "gate_proj.weight",
                            "up_proj.weight",
                            "embed_tokens.weight",
                            "lm_head.weight",
                        ]
                        col_split = any([key.endswith(filter) for filter in col_filter])

                        if col_split:
                            # state_dict[key] = concat_tensor(tensor_list, dim=0)
                            state_dict[key] = torch.cat(tensor_list, dim=0)
                            if process_exclusion:
                                # CPU 内存回收（速度很慢）
                                gc.collect()
                        elif key.endswith("o_proj.weight") or key.endswith("down_proj.weight"):
                            # state_dict[key] = concat_tensor(tensor_list, dim=1)
                            state_dict[key] = torch.cat(tensor_list, dim=1)
                            if process_exclusion:
                                # CPU 内存回收（速度很慢）
                                gc.collect()
                        if key.endswith("q_proj.weight"):
                            state_dict[key] = reshape_wq_wk(state_dict[key])
                        if key.endswith("k_proj.weight"):
                            state_dict[key] = reshape_wq_wk(state_dict[key], kv=True)
                # if tp_rank == 0:
                #     # Save gathered weights
                #     if get_pipeline_parallel_world_size() > 1:
                #         ckpt_name = f"pytorch_model-{pp_rank + 1:05d}-of-{pp_size:05d}.bin"
                #         total_size = 0
                #         weight_map = {}
                #         for name, weight in state_dict.items():
                #             weight_size = weight.numel() * dtype_byte_size(
                #                 weight.dtype
                #             )
                #             weight_map[name] = ckpt_name
                #             total_size += weight_size
                #         index_dict = dict(
                #             total_size=total_size, weight_map=weight_map
                #         )
                #         index_dicts = [None for _ in range(pp_size)]
                #         dist.gather_object(
                #             index_dict, index_dicts if pp_rank == 0 else None, group=pp_group
                #         )
                #         if pp_rank == 0:
                #             total_size = 0
                #             weight_map = {}
                #             for _index_dict in index_dicts:
                #                 total_size += _index_dict["total_size"]
                #                 weight_map.update(_index_dict["weight_map"])
                #             merged_dict = {
                #                 "metadata": {"total_size": total_size},
                #                 "weight_map": weight_map,
                #             }
                #             json.dump(
                #                 merged_dict,
                #                 open(os.path.join(path, "pytorch_model.bin.index.json"), "w"),
                #                 indent=2,
                #                 sort_keys=True,
                #             )
                #     else:
                #         ckpt_name = f"pytorch_model.bin"
                #     ckpt_path = os.path.join(path, ckpt_name)
                #     torch.save(state_dict, ckpt_path)
            if dist.is_initialized() and process_exclusion:
                dist.barrier()
        dist.barrier()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        self.save_parallel_state_dict(state_dict, config=self.config)
        return state_dict

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            **kwargs,
    ):
        state_dict = cls.load_parallel_state_dict(pretrained_model_name_or_path, *model_args, **kwargs)
        print(state_dict.keys())
        return super().from_pretrained(pretrained_model_name_or_path, state_dict=state_dict, *model_args, **kwargs)


class LlamaModelParallelPreSplitMixin(PreTrainedModelPeftMixin):
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
            *model_args,
            **kwargs,
    ):
        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_model_parallel_rank()
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def save_pretrained(
            cls,
            save_directory: Union[str, os.PathLike],
            **kwargs,
    ):
        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_model_parallel_rank()
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            save_directory = os.path.join(save_directory, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")
        super().save_pretrained(save_directory, **kwargs)


class LlamaForCausalLM(LlamaModelParallelPreSplitMixin, HfLlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModelParallel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
