"""Loguru logging setup."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logging(
    level: str = "INFO",
    log_file: str | None = "logs/hale.log",
    *,
    force: bool = False,
) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    level = level.upper()
    logger.remove()
    logger.add(sys.stderr, level=level)
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(path), level="DEBUG", rotation="10 MB")
    _CONFIGURED = True
