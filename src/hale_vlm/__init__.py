"""Hale-VLM: vision-language models on top of dense transformer LLMs."""

from hale_vlm.bootstrap import register_vlm_plugins

register_vlm_plugins()

__version__ = "0.1.0"

__all__ = ["__version__"]
