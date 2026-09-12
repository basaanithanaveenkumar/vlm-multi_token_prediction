from typing import Literal

from hale_vlm.core.config.sections.common import StrictModel


class TrainConfig(StrictModel):
    steps: int | None = 300
    epochs: int | None = None
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.0
    log_every: int = 50
    grad_clip: float = 1.0
    seed: int = 0
    checkpoint_path: str = "checkpoints/last.pt"
    checkpoint_every_epoch: bool = True
    resume: bool = True
    loss_type: Literal["ce", "focal", "label_smoothing"] = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0
    optimizer_type: str = "adamw"

    # Distributed training / data parallelism
    parallel_strategy: Literal["none", "dp", "ddp", "fsdp"] = "none"
    distributed_backend: str | None = None  # 'nccl', 'gloo', 'mpi' (auto-detected if None)
    find_unused_parameters: bool = False  # DDP: find unused parameters
    gradient_as_bucket_view: bool = True  # DDP: memory optimization
    static_graph: bool = False  # DDP: static graph optimization
