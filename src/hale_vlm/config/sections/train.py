from typing import Literal

from hale_vlm.core.config.sections.train import TrainConfig
from pydantic import Field


class VLMTrainConfig(TrainConfig):
    """Training settings extended for scratch and Hale backends."""

    backend: Literal["auto", "haleblocks", "scratch"] = "auto"
    max_epochs: int = 10
    log_tensorboard: bool = False
    log_dir: str = "./runs/hale_vlm"
    scratch_checkpoint_dir: str = "checkpoints/scratch"
