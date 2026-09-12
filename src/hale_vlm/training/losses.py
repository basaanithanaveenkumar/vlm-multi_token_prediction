"""VLM training loss and evaluation plugins."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from hale_vlm.registry import register_loss

from hale_vlm.models.scratch.factory import SCRATCH_VARIANTS
from hale_vlm.models.vlm import VLM_VARIANTS


def _hale_vlm_loss(model, batch: dict) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        labels=batch["labels"],
    )
    return outputs.loss


def _scratch_vlm_loss(model, batch: dict) -> torch.Tensor:
    images = batch.get("images")
    if images is None:
        images = batch["pixel_values"]
    outputs = model(
        images,
        batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
    )
    labels = batch["labels"]
    num_img_tokens = model.num_image_tokens
    batch_size = labels.shape[0]
    image_targets = torch.full(
        (batch_size, num_img_tokens),
        -100,
        device=labels.device,
        dtype=labels.dtype,
    )
    targets = torch.cat([image_targets, labels], dim=1)
    outputs_for_loss = outputs[:, : targets.shape[1], :]
    return F.cross_entropy(
        outputs_for_loss.reshape(-1, outputs_for_loss.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )


for _variant in VLM_VARIANTS:
    register_loss(_variant)(_hale_vlm_loss)

for _variant in SCRATCH_VARIANTS:
    register_loss(_variant)(_scratch_vlm_loss)
