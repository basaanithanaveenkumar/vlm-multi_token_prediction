"""Lightweight validation evaluator for VLM training."""

from __future__ import annotations

import torch


class VLMEvaluator:
    def run(self, model, loader) -> dict[str, float]:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                losses.append(float(outputs.loss.item()))
        model.train()
        return {"loss": sum(losses) / max(len(losses), 1)}
