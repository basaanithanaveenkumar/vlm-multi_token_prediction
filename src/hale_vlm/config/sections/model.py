from hale_core.config.sections.model import ModelConfig
from pydantic import Field

from hale_vlm.config.sections.llm import LLMConfig
from hale_vlm.config.sections.vision import VisionConfig


class VLMModelConfig(ModelConfig):
    """VLM model section: HaleBlocks model fields plus vision and LLM backbones."""

    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
