# Halo-VLM

Unified vision-language platform:

1. **Halo-VLM** — custom ViT + MoE decoder, COCO captioning, multi-token prediction research
2. **Hale-VLM** — Qwen3 / DeepSeek-R1 + SigLIP via HaleBlocks, LoRA fine-tuning, SmolVLM/VLA dataset registries

![Halo VLM](assets/halo_RB.png)

```text
src/
├── halo_vlm/    # Halo scratch models (BasicVLM, HaloVLM)
└── hale_vlm/    # HaleBlocks HF VLM stack
configs/         # Hale-VLM YAML configs
tests/
```

## Install

```bash
uv sync --extra dev
```

For Halo COCO training (Linux recommended — LAVIS/decord):

```bash
uv sync --extra dev --extra halo
```

HaleBlocks is pulled from git automatically (`pyproject.toml`).

## Halo-VLM (scratch path)

```bash
PYTHONPATH=src uv run python -m halo_vlm.train
PYTHONPATH=src uv run python -m halo_vlm.inference --help
```

## Hale-VLM (HF path)

```bash
uv run hale-vlm-train configs/qwen3_8b_overfit.yaml
uv run hale-vlm-chat configs/base.yaml --image path/to/image.jpg
```

See `configs/` for SmolVLM, SmolVLA, and robotics+VLM presets.

## Develop

```bash
pre-commit install
uv run pytest tests/ -q
uv run ruff check src tests
```

## License

MIT
