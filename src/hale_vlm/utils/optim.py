"""Optimizer builders."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def build_optimizer(name: str, params: Iterable, config: dict):
    name = name.lower()
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise KeyError(f"unknown optimizer {name!r}")
