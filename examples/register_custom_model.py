"""Extensibility: register a custom model variant without editing core code."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.registry import register_model


@register_model("demo_custom_vlm")
class DemoCustomVLM(nn.Module):
    fusion_mode = "prefix_concat"
    num_image_tokens = 1

    def __init__(self, vocab_size: int, cfg: VLMRunConfig | None = None, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.linear = nn.Linear(3, vocab_size)

    def trainable_parameters(self):
        yield from self.parameters()

    def forward(self, images, input_ids, attention_mask=None):
        del input_ids, attention_mask
        pooled = images.mean(dim=(2, 3))
        return self.linear(pooled).unsqueeze(1)


if __name__ == "__main__":
    from hale_vlm.registry import get_model

    cls = get_model("demo_custom_vlm")
    model = cls(vocab_size=128)
    out = model(torch.randn(1, 3, 224, 224), torch.zeros(1, 4, dtype=torch.long))
    print(out.shape)
