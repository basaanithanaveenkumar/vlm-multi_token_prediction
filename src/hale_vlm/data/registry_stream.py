"""Unified registry streaming for VLM and optional robotics-to-VLM conversion."""

from __future__ import annotations

from collections.abc import Iterator

from hale_vlm.config.sections.data import VLMDataConfig
from hale_vlm.data.robotics_vlm import vla_sample_to_vlm
from hale_vlm.data.sequential import SequentialMixConfig, SequentialMultiDatasetStream
from hale_vlm.data.types import RoboticsVLMMode, VLMSample
from hale_vlm.data.vla_sequential import SequentialVLAStream, VLAStreamConfig


def iter_registry_vlm_samples(
    data_cfg: VLMDataConfig,
    *,
    image_size: int,
) -> Iterator[VLMSample]:
    """Yield VLMSample from VLM registry and/or robotics registry."""
    if data_cfg.source in {"registry", "mixed_registry"}:
        stream = SequentialMultiDatasetStream(
            SequentialMixConfig(
                dataset_names=data_cfg.resolved_registry_datasets(),
                max_samples_per_dataset=data_cfg.max_samples_per_dataset,
                split=data_cfg.train_split,
                cache_dir=data_cfg.cache_dir,
                prefetch_workers=data_cfg.prefetch_workers,
                streaming=data_cfg.streaming,
                max_video_frames=data_cfg.max_video_frames,
                image_size=image_size,
            )
        )
        yield from stream

    include_vla = data_cfg.source in {"vla_registry", "mixed_registry"} or (
        data_cfg.source == "registry" and data_cfg.robotics_in_vlm_enabled()
    )
    if not include_vla:
        return

    mode = data_cfg.robotics_vlm_mode
    if mode == RoboticsVLMMode.OFF:
        mode = RoboticsVLMMode.INSTRUCTION_TUNING

    vla_stream = SequentialVLAStream(
        VLAStreamConfig(
            dataset_names=data_cfg.resolved_vla_registry_datasets(),
            max_samples_per_dataset=data_cfg.max_samples_per_dataset,
            split=data_cfg.train_split,
            cache_dir=data_cfg.cache_dir,
            prefetch_workers=data_cfg.prefetch_workers,
            streaming=data_cfg.streaming,
        )
    )
    for sample in vla_stream:
        yield vla_sample_to_vlm(sample, mode)
