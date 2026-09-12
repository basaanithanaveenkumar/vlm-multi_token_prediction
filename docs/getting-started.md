# Getting Started

```bash
uv sync --extra dev --extra test
uv run pytest tests/ -q
uv run hale-vlm-train configs/halo_moe_overfit.yaml
```

```python
from hale_vlm import load_config, build_vlm

cfg = load_config("configs/base.yaml")
model = build_vlm(cfg)
```
