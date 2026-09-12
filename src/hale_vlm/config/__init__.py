from hale_core.config import load_registered_config
from hale_core.registry import register_config

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.config.sections.model import VLMModelConfig

register_config("vlm")(VLMRunConfig)


def load_vlm_config(path: str) -> VLMRunConfig:
    cfg = load_registered_config(path, schema="vlm")
    assert isinstance(cfg, VLMRunConfig)
    return cfg


__all__ = ["VLMRunConfig", "VLMModelConfig", "load_vlm_config"]
