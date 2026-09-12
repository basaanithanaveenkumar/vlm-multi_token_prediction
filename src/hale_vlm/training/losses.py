"""VLM training loss and evaluation plugins."""

from __future__ import annotations

import torch
from hale_core.registry import register_loss

from hale_vlm.models.vlm import VLM_VARIANTS


def _vlm_loss(model, batch: dict) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        labels=batch["labels"],
    )
    return outputs.loss


for _variant in VLM_VARIANTS:
    register_loss(_variant)(_vlm_loss)
