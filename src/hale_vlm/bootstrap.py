"""Register Hale-VLM plugins with hale_core registries."""

from __future__ import annotations

_REGISTERED = False


def register_vlm_plugins() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    import hale_vlm.config  # noqa: F401
    import hale_vlm.data.datasets.builtin  # noqa: F401
    import hale_vlm.data.datasets.vla_builtin  # noqa: F401
    import hale_vlm.models.vlm  # noqa: F401
    import hale_vlm.training.losses  # noqa: F401
    import hale_vlm.training.trainer  # noqa: F401

    _REGISTERED = True
