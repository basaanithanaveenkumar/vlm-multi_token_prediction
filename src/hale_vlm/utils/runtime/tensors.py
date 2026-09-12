from __future__ import annotations

from typing import Any

import torch
from torch import nn

_UNWRAP = {"backbone", "model", "module"}


def move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def _rows_for(model: nn.Module) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []

    def add(name: str, module: nn.Module | None, param: nn.Parameter | None = None) -> None:
        if param is not None:
            n = param.numel()
            t = n if param.requires_grad else 0
            rows.append((name, n, t))
            return
        assert module is not None
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append((name, total, trainable))

    for name, param in model.named_parameters(recurse=False):
        add(name, None, param)
    for name, child in model.named_children():
        nested = list(child.named_children())
        if name in _UNWRAP and nested:
            for n, p in child.named_parameters(recurse=False):
                add(f"{name}.{n}", None, p)
            for n, g in nested:
                add(f"{name}.{n}", g)
        else:
            add(name, child)
    return rows


def format_model_summary(model: nn.Module, *, title: str | None = None) -> str:
    rows = _rows_for(model)
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    name_w, num_w = 32, 16
    bar = "-" * (name_w + num_w * 2 + 2)
    lines = []
    if title:
        lines.append(title)
    lines.append(bar)
    lines.append(f"{'Module':<{name_w}}{'Total':>{num_w}}{'Trainable':>{num_w}}")
    lines.append(bar)
    for name, tot, tr in rows:
        lines.append(f"{name:<{name_w}}{tot / 1e6:>{num_w - 1}.2f}M{tr / 1e6:>{num_w - 1}.2f}M")
    lines.append(bar)
    lines.append(
        f"{'TOTAL':<{name_w}}{total / 1e6:>{num_w - 1}.2f}M{trainable / 1e6:>{num_w - 1}.2f}M"
    )
    lines.append(bar)
    return "\n".join(lines)


def log_model_summary(model: nn.Module, *, title: str | None = None) -> str:
    from loguru import logger

    text = format_model_summary(model, title=title)
    logger.info("\n{}", text)
    return text
