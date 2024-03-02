from typing import Callable
from general_util import training_utils
from general_util.dist_utils import print_rank_0
from general_util.logger import get_child_logger
from general_util.training_utils import get_zero_stage
from general_util.transformer_engine import convert_model

logger = get_child_logger(__name__)

try:
    import transformer_engine.pytorch as transformer_engine
    from transformer_engine.common import recipe
except ImportError:
    logger.info("Transformer Engine package is missing, skipping tests")


def fp8_func_wrap(func: Callable, fp8_flag: bool, fp8_recipe, *args, **kwargs):
    if fp8_flag:
        with transformer_engine.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return func(*args, **kwargs)
    else:
        return func(*args, **kwargs)
