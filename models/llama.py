from transformers.models.llama import LlamaPreTrainedModel
from typing import Optional, Union
import os
from logging import Logger

from general_util.logger import get_child_logger

logger: Logger = get_child_logger(__name__)


class LlamaPreTrainedModelPeftMixin(LlamaPreTrainedModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        gradient_checkpointing = kwargs.pop("gradient_checkpointing", False)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()

        return model
