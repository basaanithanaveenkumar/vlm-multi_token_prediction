"""Interactive image+text chat CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from hale_vlm.config import load_vlm_config
from hale_vlm.inference.scratch import VLMInference
from hale_vlm.models.vlm import build_vlm


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def _chat_hale(cfg, args, device) -> None:
    model = build_vlm(cfg).to(device)
    model.eval()

    tokenizer = model.tokenizer
    image_token = cfg.model.llm.image_token
    user_prompt = f"User: {image_token}\n{args.prompt}\nAssistant:"
    encoded = tokenizer(user_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    pixel_values = _load_image(args.image, cfg.model.vision.image_size).to(device)

    with torch.no_grad():
        text_embeds = model.llm.embed_tokens(input_ids)
        image_embeds = model.encode_images(pixel_values)
        inputs_embeds, attention_mask = model.merge_image_embeddings(
            input_ids,
            text_embeds,
            image_embeds,
        )
        outputs = model.llm.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def _chat_scratch(cfg, args, device) -> None:
    from transformers import AutoTokenizer

    if args.checkpoint is None:
        raise ValueError("Scratch chat requires --checkpoint pointing to a .pt file")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.scratch.tokenizer_id)
    engine = VLMInference(
        model_path=str(args.checkpoint),
        device=str(device),
        tokenizer=tokenizer,
        max_length=cfg.model.max_length,
    )
    image = engine.load_image(str(args.image))
    prompt = args.prompt or "Describe the image"
    init_tokens = engine.prepare_initial_tokens(f"Describe the image{prompt}")
    generated_ids, decoded = engine.generate_greedy(
        image,
        init_tokens,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    del generated_ids
    print(decoded[0] if decoded else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a VLM model")
    parser.add_argument("config", type=Path, help="Path to YAML config")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Scratch checkpoint (.pt). Required for basic/halo_moe architectures.",
    )
    args = parser.parse_args()

    cfg = load_vlm_config(str(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architecture = cfg.model.resolved_architecture(cfg.variant)
    if architecture == "hale":
        _chat_hale(cfg, args, device)
    else:
        _chat_scratch(cfg, args, device)


if __name__ == "__main__":
    main()
