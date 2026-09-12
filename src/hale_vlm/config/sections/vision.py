from typing import Literal

from hale_core.config.sections.common import StrictModel


class VisionConfig(StrictModel):
    """Vision tower and projector settings."""

    encoder: Literal["siglip", "clip"] = "siglip"
    model_id: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    freeze_encoder: bool = True
    projector_type: Literal["mlp", "linear"] = "mlp"
    projector_hidden_dim: int | None = None
    projector_dropout: float = 0.0
    num_image_tokens: int = 256
