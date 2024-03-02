import gc
import json
import os
from collections import OrderedDict
from glob import glob
from typing import Union, Optional, Callable, List, Tuple

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from einops import rearrange
import omegaconf
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
    LlamaPreTrainedModel,
    CausalLMOutputWithPast,
)

from general_util.dist_utils import get_pipeline_parallel_rank, get_pipeline_parallel_world_size
from general_util.logger import get_child_logger
from models.llama import PreTrainedModelPeftMixin, llama_last_token_forward_value, llama_dpo_batch_forward
from models.utils import DPOModelOutput, RewardModelOutput

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


REFERENCE_MODEL: LlamaPreTrainedModel


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

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            *args,
            **kwargs,
    ):
        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_model_parallel_rank()
            save_directory = os.path.join(save_directory, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")
        super().save_pretrained(save_directory, *args, **kwargs)

    @classmethod
    def from_pretrained_with_ref_model(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], ref_model: LlamaPreTrainedModel,
                                       *model_args, **kwargs):
        global REFERENCE_MODEL
        REFERENCE_MODEL = ref_model
        REFERENCE_MODEL.eval()
        REFERENCE_MODEL.to(device=torch.cuda.current_device())

        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model


class LlamaForCausalLM(LlamaModelParallelPreSplitMixin, HfLlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaModelParallel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # if self.config.pretraining_tp > 1:
        #     lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        #     logits = [nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        #     logits = torch.cat(logits, dim=-1)
        # else:
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_labels[shift_labels.eq(self.config.pad_token_id)] = -100  # Take care of here.
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class LlamaModelForSequenceClassificationForRL(LlamaModelParallelPreSplitMixin, LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, reduction_ids: List[int]):
        super().__init__(config)
        self.model = LlamaModelParallel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        if isinstance(reduction_ids, omegaconf.ListConfig):
            reduction_ids = list(reduction_ids)

        self.reduction_ids = reduction_ids

    def logit2prob(self, logits):
        probs = torch.softmax(logits, dim=-1)
        if len(logits.size()) == 3:
            probs = probs[:, :, self.reduction_ids].sum(dim=-1)
        elif len(logits.size()) == 2:
            probs = probs[:, self.reduction_ids].sum(dim=-1)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.size()}")
        return probs

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[Tuple, RewardModelOutput]:
        values, rewards, sequence_lengths = llama_last_token_forward_value(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id)
        values = self.logit2prob(values)
        rewards = self.logit2prob(rewards)

        value_mask = input_ids.eq(self.config.pad_token_id)
        values = values.masked_fill(value_mask, 0)

        return RewardModelOutput(
            values=values,
            chosen_end_scores=rewards,
            sequence_lengths=sequence_lengths,
        )


class LlamaForCausalLMDPO(LlamaForCausalLM):
    def __init__(self, config, beta: float = 0.1, label_smoothing: float = 0.0, use_ipo: bool = False, loss_type: str = "sigmoid"):
        super().__init__(config)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.use_ipo = use_ipo
        self.loss_type = loss_type
        logger.warning(f"Using loss type: {self.loss_type}")

        # Initialize weights and apply final processing
        self.post_init()

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.use_ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "sigmoid":
            log_sigmoid = nn.LogSigmoid()
            losses = -log_sigmoid(self.beta * logits) * (1 - self.label_smoothing) - log_sigmoid(-self.beta * logits) * self.label_smoothing
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses.mean(), chosen_rewards, rejected_rewards

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        half = input_ids.size(0) // 2

        policy_logits, policy_logprobs, policy_loss_mask = llama_dpo_batch_forward(self, input_ids, attention_mask, labels)
        with torch.no_grad():
            ref_logits, ref_logprobs, ref_loss_mask = llama_dpo_batch_forward(REFERENCE_MODEL, input_ids, attention_mask, labels)

        policy_chosen_logits, policy_reject_logits = policy_logits[:half], policy_logits[half:]
        policy_chosen_logprobs, policy_reject_logprobs = policy_logprobs[:half], policy_logprobs[half:]

        # ref_chosen_logits, ref_reject_logits = ref_logits[:half], ref_logits[half:]
        ref_chosen_logprobs, ref_reject_logprobs = ref_logprobs[:half], ref_logprobs[half:]

        loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps=policy_chosen_logprobs,
            policy_rejected_logps=policy_reject_logprobs,
            reference_chosen_logps=ref_chosen_logprobs,
            reference_rejected_logps=ref_reject_logprobs,
            reference_free=False,
        )

        return DPOModelOutput(
            loss=loss,
            chosen_reward=chosen_rewards.mean(),
            rejected_reward=rejected_rewards.mean(),
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_reject_logits,
        )

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            save_peft_format: bool = True,
            **kwargs,
    ):
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token,
                                **kwargs)

        if is_main_process:
            config = self.config
            config.architectures = ["LlamaForCausalLM"]
            config.save_pretrained(save_directory)
            logger.warning("Config architecture is override to LlamaForCausalLM")
