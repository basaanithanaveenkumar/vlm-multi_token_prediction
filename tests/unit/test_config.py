from pathlib import Path

import pytest

from hale_vlm.config import load_vlm_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.smoke
def test_load_base_config():
    cfg = load_vlm_config(CONFIGS / "base.yaml")
    assert cfg.variant == "qwen3_8b_vlm"
    assert cfg.model.llm.backbone == "qwen3-8b"
    assert cfg.model.vision.encoder == "siglip"


@pytest.mark.smoke
def test_load_deepseek_config_inheritance():
    cfg = load_vlm_config(CONFIGS / "deepseek_r1_qwen_7b.yaml")
    assert cfg.variant == "deepseek_r1_qwen_7b_vlm"
    assert cfg.model.llm.backbone == "deepseek-r1-qwen-7b"
    assert cfg.model.llm.reasoning_mode is True
    assert cfg.model.vision.freeze_encoder is True
