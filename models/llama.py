import os
from dataclasses import dataclass
from logging import Logger
from typing import Optional, Union, Tuple, List, Callable

import hydra
import omegaconf
import torch
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    TaskType,
    get_peft_model,
)
from peft.tuners.lora import LoraLayer
from torch import nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as HfLlamaForCausalLM,
    PreTrainedModel,
    CausalLMOutputWithPast
)

from general_util.logger import get_child_logger
from general_util.training_utils import get_rank

logger: Logger = get_child_logger(__name__)

REFERENCE_MODEL: HfLlamaForCausalLM


@dataclass
class DPOModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    chosen_reward: torch.FloatTensor = None
    rejected_reward: torch.FloatTensor = None
    policy_chosen_logits: torch.FloatTensor = None
    policy_rejected_logits: torch.FloatTensor = None


def return_single_device_map():
    return {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}


class PreTrainedModelPeftMixin(PreTrainedModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

        return model

    @classmethod
    def from_pretrained_with_ref_model(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], ref_model: PreTrainedModel,
                                       *model_args, **kwargs):
        global REFERENCE_MODEL
        REFERENCE_MODEL = ref_model
        REFERENCE_MODEL.eval()

        model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    @classmethod
    def from_pretrained_with_ref_model_lora(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        lora_config = kwargs.pop("lora_config", None)
        assert lora_config is not None, "lora_config must be provided to enable lora training."

        base_model = cls.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        global REFERENCE_MODEL
        REFERENCE_MODEL = base_model

        enable_quantization = "quantization_config" in kwargs

        if lora_config is None:
            lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32,
                                     lora_dropout=0.1)

        logger.warning(lora_config)
        logger.info(lora_config.target_modules.__class__)
        if isinstance(lora_config.target_modules, omegaconf.listconfig.ListConfig):
            lora_config.target_modules = list(lora_config.target_modules)
        elif isinstance(lora_config.target_modules, omegaconf.DictConfig):
            lora_config.target_modules = hydra.utils.instantiate(lora_config.target_modules, model=base_model)
        else:
            raise ValueError(f"Unsupported type of target modules: {lora_config.target_modules.__class__}")

        if isinstance(lora_config.modules_to_save, omegaconf.listconfig.ListConfig):
            lora_config.modules_to_save = list(lora_config.modules_to_save)

        logger.info(lora_config.target_modules.__class__)
        logger.warning(lora_config.target_modules)

        gradient_checkpointing = base_model.model.gradient_checkpointing
        if enable_quantization:
            logger.warning(f"Rank {get_rank()} is being loaded with quantization.")
            logger.info(f"Quantization config: {kwargs['quantization_config']}")
            base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=gradient_checkpointing)

        model = get_peft_model(base_model, lora_config)

        logger.info(f"Reference model type: {REFERENCE_MODEL.__class__.__name__}")
        logger.info(f"Actor model type: {model.__class__.__name__}")

        compute_dtype = kwargs["torch_dtype"]
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if compute_dtype == torch.bfloat16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if compute_dtype and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

        model.print_trainable_parameters()

        logger.info(f"Config pad token id after loading pre-trained weights: {model.config.pad_token_id}")
        logger.info(model.lm_head.__class__.__name__)

        return model


def llama_dpo_batch_forward(model: HfLlamaForCausalLM, input_ids: torch.LongTensor, attention_mask: torch.Tensor, labels: torch.LongTensor):
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)
    logits = logits.float()

    labels = labels[:, 1:].clone()

    loss_mask = labels.ne(model.config.pad_token_id)
    labels[~loss_mask] = 0

    per_token_logprobs = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return logits, (per_token_logprobs * loss_mask).sum(-1), loss_mask


class LlamaForCausalLMDPO(PreTrainedModelPeftMixin, HfLlamaForCausalLM):
    def __init__(self, config, beta: float = 0.1):
        super().__init__(config)
        self.beta = beta

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

        losses = -nn.functional.logsigmoid(self.beta * logits)
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


class LlamaForCausalLM(PreTrainedModelPeftMixin, HfLlamaForCausalLM):
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
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [nn.functional.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
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
            shift_labels[shift_labels.eq(self.config.pad_token_id)] = -100
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
