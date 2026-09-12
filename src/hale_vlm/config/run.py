"""VLM run config extending HaleBlocks RunConfig."""

from hale_core.config.run import RunConfig
from pydantic import Field

from hale_vlm.config.sections.data import VLMDataConfig
from hale_vlm.config.sections.model import VLMModelConfig


class VLMRunConfig(RunConfig):
    """Full configuration for vision-language model training and inference."""

    model: VLMModelConfig = Field(default_factory=VLMModelConfig)
    data: VLMDataConfig = Field(default_factory=VLMDataConfig)
