"""VLM run config extending the core RunConfig."""

from typing import Literal

from pydantic import Field

from hale_vlm.config.sections.data import VLMDataConfig
from hale_vlm.config.sections.model import VLMModelConfig
from hale_vlm.config.sections.train import VLMTrainConfig
from hale_vlm.core.config.run import RunConfig


class VLMRunConfig(RunConfig):
    """Full configuration for vision-language model training and inference."""

    model: VLMModelConfig = Field(default_factory=VLMModelConfig)
    train: VLMTrainConfig = Field(default_factory=VLMTrainConfig)
    data: VLMDataConfig = Field(default_factory=VLMDataConfig)

    def resolved_train_backend(self) -> Literal["haleblocks", "scratch"]:
        if self.train.backend != "auto":
            return self.train.backend
        architecture = self.model.resolved_architecture(self.variant)
        return "scratch" if architecture in {"basic", "halo_moe"} else "haleblocks"
