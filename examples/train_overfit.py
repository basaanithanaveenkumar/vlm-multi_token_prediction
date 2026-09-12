"""Library usage: load config and build a model."""

from hale_vlm import build_vlm, load_config


def main() -> None:
    cfg = load_config("configs/halo_moe_overfit.yaml")
    model = build_vlm(cfg)
    print(type(model).__name__, "num_image_tokens=", model.num_image_tokens)


if __name__ == "__main__":
    main()
