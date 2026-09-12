"""Multimodal dataset and dataloader utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from transformers import AutoTokenizer

from hale_vlm.config.run import VLMRunConfig
from hale_vlm.data.registry_stream import iter_registry_vlm_samples
from hale_vlm.data.types import Modality, VLMSample
from hale_vlm.llm.backbones import resolve_llm_config


@dataclass
class MultimodalSample:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    modality: str
    dataset: str


class MultimodalDataset(Dataset):
    """Image + caption pairs with prompt templating."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        tokenizer,
        image_token: str,
        image_size: int,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.max_length = max_length
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MultimodalSample:
        row = self.records[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        pixel_values = self.transform(image)

        caption = row["text"]
        prompt = f"User: {self.image_token}\nDescribe the image.\nAssistant: {caption}"
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return MultimodalSample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            modality="image",
            dataset=row.get("dataset", "local"),
        )


class EncodedRegistryDataset(Dataset):
    """Materialize a bounded slice from the sequential registry stream for map-style training."""

    def __init__(self, cfg: VLMRunConfig, tokenizer) -> None:
        self.samples = list(RegistryStreamingDataset(cfg, tokenizer))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultimodalSample:
        return self.samples[idx]


class RegistryStreamingDataset(IterableDataset):
    """Stream all registered datasets sequentially with bounded sample counts."""

    def __init__(self, cfg: VLMRunConfig, tokenizer) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_token = cfg.model.llm.image_token
        self.max_length = cfg.model.max_length
        self.image_size = cfg.model.vision.image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __iter__(self) -> Iterator[MultimodalSample]:
        data_cfg = self.cfg.data
        for sample in iter_registry_vlm_samples(
            data_cfg,
            image_size=self.image_size,
        ):
            yield self._encode(sample)

    def _encode(self, sample: VLMSample) -> MultimodalSample:
        pixel_values = self._visual_tensor(sample)
        prompt = self._build_prompt(sample)
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return MultimodalSample(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            modality=sample.modality.value,
            dataset=sample.dataset,
        )

    def _build_prompt(self, sample: VLMSample) -> str:
        if sample.modality == Modality.TEXT:
            return sample.text
        if sample.modality == Modality.VIDEO:
            return f"User: {self.image_token}\n{sample.text}\nAssistant: {sample.text}"
        if sample.modality == Modality.MULTI_IMAGE:
            tokens = " ".join(self.image_token for _ in sample.images) or self.image_token
            return f"User: {tokens}\n{sample.text}\nAssistant: {sample.text}"
        return f"User: {self.image_token}\n{sample.text}\nAssistant: {sample.text}"

    def _visual_tensor(self, sample: VLMSample) -> torch.Tensor:
        frames = sample.visual_frames
        if not frames:
            return torch.zeros(3, self.image_size, self.image_size)
        tensors = [self.transform(frame) for frame in frames]
        if sample.modality == Modality.VIDEO and len(tensors) > 1:
            return torch.stack(tensors)
        return tensors[0]


def _collate(samples: list[MultimodalSample]) -> dict[str, torch.Tensor | list[str]]:
    pixel_values = torch.stack(
        [s.pixel_values if s.pixel_values.ndim == 3 else s.pixel_values[0] for s in samples]
    )
    return {
        "pixel_values": pixel_values,
        "input_ids": torch.stack([s.input_ids for s in samples]),
        "attention_mask": torch.stack([s.attention_mask for s in samples]),
        "labels": torch.stack([s.labels for s in samples]),
        "modality": [s.modality for s in samples],
        "dataset": [s.dataset for s in samples],
    }


class MultimodalDataModule:
    """HaleBlocks-compatible data module for VLM training."""

    def __init__(self, cfg: VLMRunConfig, tokenizer=None) -> None:
        self.cfg = cfg
        if tokenizer is None:
            llm_cfg = resolve_llm_config(cfg.model.llm)
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_cfg.model_id,
                trust_remote_code=llm_cfg.trust_remote_code,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        self._train: Dataset | IterableDataset | None = None
        self._val: Dataset | None = None

    def _load_split(self, split: str) -> list[dict[str, Any]]:
        data_cfg = self.cfg.data
        if data_cfg.source == "overfit":
            text = data_cfg.overfit_text or "A red square on a white background."
            size = self.cfg.model.vision.image_size
            image = Image.new("RGB", (size, size), "white")
            copies = data_cfg.n_overfit_copies if split == data_cfg.train_split else 4
            return [
                {"image": image.copy(), "text": text, "dataset": "overfit"} for _ in range(copies)
            ]

        ds = load_dataset(
            data_cfg.dataset,
            data_cfg.subset,
            split=split,
            cache_dir=data_cfg.cache_dir,
        )
        records = []
        for row in ds:
            image = row.get("image")
            text = row.get(data_cfg.text_field) or row.get("caption") or row.get("text")
            if image is None or text is None:
                continue
            records.append({"image": image, "text": text, "dataset": data_cfg.dataset})
            limit = data_cfg.train_size if split == data_cfg.train_split else data_cfg.val_size
            if limit is not None and len(records) >= limit:
                break
        return records

    def train_loader(self) -> DataLoader:
        if self._train is None:
            if self.cfg.data.source in {"registry", "vla_registry", "mixed_registry"}:
                self._train = EncodedRegistryDataset(self.cfg, self.tokenizer)
            else:
                records = self._load_split(self.cfg.data.train_split)
                self._train = MultimodalDataset(
                    records,
                    tokenizer=self.tokenizer,
                    image_token=self.cfg.model.llm.image_token,
                    image_size=self.cfg.model.vision.image_size,
                    max_length=self.cfg.model.max_length,
                )
        return DataLoader(
            self._train,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.data.source not in {"registry", "vla_registry", "mixed_registry"},
            collate_fn=_collate,
        )

    def val_loader(self, train_loader: DataLoader | None = None) -> DataLoader:
        del train_loader
        if self._val is None:
            records = self._load_split(self.cfg.data.val_split)
            self._val = MultimodalDataset(
                records,
                tokenizer=self.tokenizer,
                image_token=self.cfg.model.llm.image_token,
                image_size=self.cfg.model.vision.image_size,
                max_length=self.cfg.model.max_length,
            )
        return DataLoader(
            self._val,
            batch_size=self.cfg.train.batch_size,
            shuffle=False,
            collate_fn=_collate,
        )

    def viz_prompt(self, step: int) -> str:
        del step
        return f"User: {self.cfg.model.llm.image_token}\nDescribe the image.\nAssistant:"
