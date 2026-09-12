"""Tests for unified VLM package merge."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hale_vlm.config import load_vlm_config
from hale_vlm.models.scratch.halo_vlm import HaloVLM
from hale_vlm.models.vlm import build_vlm

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.smoke
def test_load_scratch_configs():
    halo_cfg = load_vlm_config(CONFIGS / "halo_moe_coco.yaml")
    assert halo_cfg.variant == "halo_vlm_moe"
    assert halo_cfg.model.resolved_architecture(halo_cfg.variant) == "halo_moe"
    assert halo_cfg.resolved_train_backend() == "scratch"

    basic_cfg = load_vlm_config(CONFIGS / "basic_vlm_coco.yaml")
    assert basic_cfg.variant == "basic_vlm_scratch"
    assert basic_cfg.model.resolved_architecture(basic_cfg.variant) == "basic"


@pytest.mark.smoke
def test_build_scratch_models():
    halo_cfg = load_vlm_config(CONFIGS / "halo_moe_coco.yaml")
    halo_model = build_vlm(halo_cfg)
    assert isinstance(halo_model, HaloVLM)
    assert halo_model.num_image_tokens == 196

    basic_cfg = load_vlm_config(CONFIGS / "basic_vlm_coco.yaml")
    pytest.importorskip("open_clip")
    from hale_vlm.models.scratch.basic_vlm import BasicVLM

    basic_model = build_vlm(basic_cfg)
    assert isinstance(basic_model, BasicVLM)
    assert basic_model.fusion_mode == "prefix_concat"
    assert basic_model.num_image_tokens == 1


@pytest.mark.smoke
def test_halo_moe_forward_pass():
    cfg = load_vlm_config(CONFIGS / "halo_moe_coco.yaml")
    model = build_vlm(cfg)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, cfg.model.scratch.vocab_size, (2, 8))
    logits = model(images, input_ids, attention_mask=torch.ones(2, 8))
    expected_seq = model.num_image_tokens + input_ids.shape[1]
    assert logits.shape == (2, expected_seq, cfg.model.scratch.vocab_size)


@pytest.mark.smoke
def test_basic_vlm_forward_pass():
    pytest.importorskip("open_clip")
    from hale_vlm.models.scratch.basic_vlm import BasicVLM

    cfg = load_vlm_config(CONFIGS / "basic_vlm_coco.yaml")
    model = build_vlm(cfg)
    assert isinstance(model, BasicVLM)
    images = torch.randn(2, 3, 224, 224)
    input_ids = torch.randint(0, cfg.model.scratch.vocab_size, (2, 8))
    attention_mask = torch.ones(2, 8)
    logits = model(images, input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 1 + input_ids.shape[1], cfg.model.scratch.vocab_size)


@pytest.mark.smoke
def test_hale_config_still_uses_haleblocks_backend():
    cfg = load_vlm_config(CONFIGS / "base.yaml")
    assert cfg.resolved_train_backend() == "haleblocks"
    assert cfg.model.resolved_architecture(cfg.variant) == "hale"

