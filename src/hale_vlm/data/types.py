"""Shared types for VLM dataset registry and streaming loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from PIL import Image


class Modality(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    MULTI_IMAGE = "multi_image"


class TrainingStage(StrEnum):
    """SmolVLM paper training stages."""

    VISION = "vision"
    VIDEO = "video"
    CONTEXT = "context"
    TEXT_SFT = "text_sft"
    REJECTED = "rejected"


class VisionCategory(StrEnum):
    """Vision-stage mixture buckets (Figure 8, SmolVLM paper)."""

    OCR_DOCUMENTS = "ocr_documents"
    CAPTIONING = "captioning"
    CHART_UNDERSTANDING = "chart_understanding"
    REASONING_LOGIC = "reasoning_logic"
    TABLE_UNDERSTANDING = "table_understanding"
    VISUAL_QA = "visual_qa"
    MULTI_IMAGE = "multi_image"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MATH_HANDWRITING = "math_handwriting"


class VideoCategory(StrEnum):
    """Video fine-tuning buckets (Figure 8, SmolVLM paper)."""

    VISUAL_DESCRIPTION = "visual_description"
    TEMPORAL_UNDERSTANDING = "temporal_understanding"
    NARRATIVE = "narrative"
    MULTI_IMAGE = "multi_image"
    TEXT_SFT = "text_sft"


@dataclass(frozen=True)
class DatasetSpec:
    """Static metadata for a registered VLM dataset."""

    name: str
    hf_path: str
    modality: Modality
    stage: TrainingStage
    description: str = ""
    category: str | None = None
    paper_reference: str = ""
    enabled: bool = True
    config_name: str | None = None
    default_split: str = "train"
    image_fields: tuple[str, ...] = ("image", "images")
    video_fields: tuple[str, ...] = ("video", "video_path", "video_id")
    text_fields: tuple[str, ...] = ("text", "caption", "conversations", "question", "answer")
    conversation_field: str | None = "conversations"
    streaming: bool = True
    trust_remote_code: bool = False


@dataclass
class VLMSample:
    """Normalized training sample across image, video, and text datasets."""

    dataset: str
    modality: Modality
    text: str
    stage: TrainingStage | None = None
    category: str | None = None
    images: list[Image.Image] = field(default_factory=list)
    video_frames: list[Image.Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        return self.modality in {Modality.IMAGE, Modality.VIDEO, Modality.MULTI_IMAGE}

    @property
    def visual_frames(self) -> list[Image.Image]:
        if self.modality == Modality.VIDEO:
            return self.video_frames
        return self.images


BatchModality = Literal["image", "video", "text", "multi_image"]


class VLAStage(StrEnum):
    """SmolVLA paper dataset stages."""

    COMMUNITY = "community"
    SIMULATION = "simulation"
    REAL_WORLD = "real_world"


class RobotEmbodiment(StrEnum):
    """Robot platforms referenced in SmolVLA."""

    SO100 = "so100"
    SO101 = "so101"
    PANDA = "panda"
    SAWYER = "sawyer"
    MIXED = "mixed"


class RoboticsVLMMode(StrEnum):
    """How robotics registry samples are exposed to VLM training."""

    OFF = "off"
    PRETRAINING = "pretraining"
    FINETUNING = "finetuning"
    INSTRUCTION_TUNING = "instruction_tuning"


@dataclass(frozen=True)
class VLADatasetSpec:
    """Static metadata for a registered VLA / robotics dataset."""

    name: str
    hf_path: str
    stage: VLAStage
    embodiment: RobotEmbodiment = RobotEmbodiment.SO100
    description: str = ""
    paper_reference: str = "SmolVLA (arXiv:2506.01844)"
    enabled: bool = True
    config_name: str | None = None
    default_split: str = "train"
    task_fields: tuple[str, ...] = ("task", "language_instruction", "prompt", "instruction")
    action_fields: tuple[str, ...] = ("action", "actions")
    state_fields: tuple[str, ...] = ("observation.state", "state", "proprio")
    image_fields: tuple[str, ...] = ("image", "images")
    camera_prefix: str = "observation.images"
    streaming: bool = True
    trust_remote_code: bool = False
    episodes: int | None = None


@dataclass
class VLASample:
    """Normalized robotics demonstration frame."""

    dataset: str
    stage: VLAStage
    task: str
    images: list[Image.Image] = field(default_factory=list)
    action: list[float] | None = None
    state: list[float] | None = None
    embodiment: RobotEmbodiment = RobotEmbodiment.SO100
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_visual(self) -> bool:
        return bool(self.images)
