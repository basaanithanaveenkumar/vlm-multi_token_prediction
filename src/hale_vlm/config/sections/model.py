from typing import Literal

from hale_vlm.core.config.sections.model import ModelConfig
from pydantic import Field

from hale_vlm.config.sections.llm import LLMConfig
from hale_vlm.config.sections.scratch import ScratchConfig
from hale_vlm.config.sections.vision import VisionConfig

_VARIANT_ARCHITECTURE = {
    "qwen3_8b_vlm": "hale",
    "deepseek_r1_qwen_7b_vlm": "hale",
    "basic_vlm_scratch": "basic",
    "halo_vlm_moe": "halo_moe",
}


class VLMModelConfig(ModelConfig):
    """VLM model section: HaleBlocks model fields plus vision and LLM backbones."""

    architecture: Literal["auto", "hale", "basic", "halo_moe"] = "auto"
    vision: VisionConfig = Field(default_factory=VisionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scratch: ScratchConfig = Field(default_factory=ScratchConfig)

    def resolved_architecture(self, variant: str) -> Literal["hale", "basic", "halo_moe"]:
        if self.architecture != "auto":
            return self.architecture
        return _VARIANT_ARCHITECTURE.get(variant, "hale")
