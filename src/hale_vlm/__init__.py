"""Hale-VLM: vision-language modeling library."""

from hale_vlm._version import __version__
from hale_vlm.bootstrap import register_vlm_plugins
from hale_vlm.config import VLMRunConfig, load_config, load_vlm_config
from hale_vlm.models import ALL_VLM_VARIANTS, build_vlm
from hale_vlm.registry import register_model

register_vlm_plugins()


__all__ = [
    "__version__",
    "ALL_VLM_VARIANTS",
    "VLMRunConfig",
    "build_vlm",
    "load_config",
    "load_vlm_config",
    "register_model",
]
