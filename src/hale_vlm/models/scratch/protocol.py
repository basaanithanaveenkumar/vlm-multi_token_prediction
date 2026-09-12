"""Shared interface for Hale and scratch VLM backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, Protocol

import torch
from torch.nn import Parameter


class VLMProtocol(Protocol):
    """Minimal contract used by unified training and inference dispatch."""

    fusion_mode: Literal["token_replace", "prefix_concat"]
    num_image_tokens: int

    def trainable_parameters(self) -> Iterator[Parameter]: ...

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
