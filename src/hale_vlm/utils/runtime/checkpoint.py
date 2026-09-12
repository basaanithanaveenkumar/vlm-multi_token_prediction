from __future__ import annotations

import os
from typing import Any

import torch
from loguru import logger


class CheckpointStore:
    """Persist and restore training state. Callers depend on this class, not torch.save."""

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def save(self, path: str, **state: Any) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(state, path)
        logger.info("saved checkpoint {}", path)
        logger.debug("checkpoint keys={}", list(state))

    def load(self, path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        if not self.exists(path):
            logger.error("checkpoint not found: {}", path)
            raise FileNotFoundError(path)
        logger.info("loading checkpoint {} map_location={}", path, map_location)
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        logger.debug("loaded checkpoint keys={}", list(ckpt))
        return ckpt


_DEFAULT_STORE = CheckpointStore()


def save_checkpoint(path: str, **state: Any) -> None:
    _DEFAULT_STORE.save(path, **state)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return _DEFAULT_STORE.load(path, map_location)
