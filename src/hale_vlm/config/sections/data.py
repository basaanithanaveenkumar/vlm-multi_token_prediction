"""VLM-specific data configuration."""

from __future__ import annotations

from typing import Literal

from hale_core.config.sections.data import DataConfig
from pydantic import Field

from hale_vlm.data.types import RoboticsVLMMode


class VLMDataConfig(DataConfig):
    source: Literal["huggingface", "overfit", "registry", "vla_registry", "mixed_registry"] = (
        "huggingface"
    )
    registry_stage: Literal["all", "vision", "video", "context"] = "all"
    registry_datasets: list[str] = Field(default_factory=list)
    # SmolVLA robotics registry (can also feed VLM training via robotics_vlm_mode)
    vla_registry_stage: Literal["all", "community", "simulation", "real_world"] = "all"
    vla_registry_datasets: list[str] = Field(default_factory=list)
    robotics_vlm_mode: RoboticsVLMMode = RoboticsVLMMode.OFF
    max_samples_per_dataset: int | None = 256
    streaming: bool = True
    prefetch_workers: int = 2
    max_video_frames: int = 8

    def model_post_init(self, __context) -> None:
        if not self.registry_datasets:
            from hale_vlm.data.catalog import SMOLVLM_ALL_DATASETS

            self.registry_datasets = list(SMOLVLM_ALL_DATASETS)
        if not self.vla_registry_datasets:
            from hale_vlm.data.vla_catalog import SMOLVLA_ALL_DATASETS

            self.vla_registry_datasets = list(SMOLVLA_ALL_DATASETS)

    def resolved_registry_datasets(self) -> list[str]:
        from hale_vlm.data.catalog import (
            SMOLVLM_ALL_DATASETS,
            SMOLVLM_CONTEXT_DATASETS,
            SMOLVLM_VIDEO_DATASETS,
            SMOLVLM_VISION_DATASETS,
        )

        presets = {
            "vision": SMOLVLM_VISION_DATASETS,
            "video": SMOLVLM_VIDEO_DATASETS,
            "context": SMOLVLM_CONTEXT_DATASETS,
            "all": SMOLVLM_ALL_DATASETS,
        }
        if self.registry_stage != "all":
            return list(presets[self.registry_stage])
        return self.registry_datasets

    def resolved_vla_registry_datasets(self) -> list[str]:
        from hale_vlm.data.vla_catalog import (
            SMOLVLA_ALL_DATASETS,
            SMOLVLA_COMMUNITY_DATASETS,
            SMOLVLA_REAL_WORLD_DATASETS,
            SMOLVLA_SIMULATION_DATASETS,
        )

        presets = {
            "community": SMOLVLA_COMMUNITY_DATASETS,
            "simulation": SMOLVLA_SIMULATION_DATASETS,
            "real_world": SMOLVLA_REAL_WORLD_DATASETS,
            "all": SMOLVLA_ALL_DATASETS,
        }
        if self.vla_registry_stage != "all":
            return list(presets[self.vla_registry_stage])
        return self.vla_registry_datasets

    def robotics_in_vlm_enabled(self) -> bool:
        return self.robotics_vlm_mode != RoboticsVLMMode.OFF

    def uses_registry_stream(self) -> bool:
        return self.source in {"registry", "mixed_registry"} or self.robotics_in_vlm_enabled()
