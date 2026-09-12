"""Register hale-vlm plugins with local registries."""

from __future__ import annotations

_REGISTERED = False


def register_vlm_plugins() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import hale_vlm.utils.logging  # noqa: F401 — loggers
    import hale_vlm.training.base  # noqa: F401 — default trainer
    import hale_vlm.config  # noqa: F401 — vlm config schema
    import hale_vlm.data.datasets.builtin  # noqa: F401
    import hale_vlm.data.datasets.vla_builtin  # noqa: F401
    import hale_vlm.models.scratch.factory  # noqa: F401
    import hale_vlm.models.vlm  # noqa: F401
    import hale_vlm.training.losses  # noqa: F401
    import hale_vlm.training.trainer  # noqa: F401

    _REGISTERED = True
