from hale_vlm.core.config.load import load_registered_config
from hale_vlm.registry import register_config

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.config.sections.model import VLMModelConfig

register_config("vlm")(VLMRunConfig)


def load_vlm_config(path: str) -> VLMRunConfig:
    cfg = load_registered_config(path, schema="vlm")
    assert isinstance(cfg, VLMRunConfig)
    return cfg


load_config = load_vlm_config

__all__ = ["VLMRunConfig", "VLMModelConfig", "load_vlm_config", "load_config"]
