"""Scratch-trained VLM models (BasicVLM, HaloVLM)."""

from hale_vlm.models.scratch.factory import SCRATCH_VARIANTS, build_scratch_vlm

__all__ = ["SCRATCH_VARIANTS", "build_scratch_vlm"]


def __getattr__(name: str):
    if name == "BasicVLM":
        from hale_vlm.models.scratch.basic_vlm import BasicVLM

        return BasicVLM
    if name == "HaloVLM":
        from hale_vlm.models.scratch.halo_vlm import HaloVLM

        return HaloVLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
