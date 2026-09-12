"""VLM trainer that optimizes only fine-tuning parameters (LoRA + projector)."""

from __future__ import annotations

from hale_vlm.utils import optim as optim_module
from hale_vlm.registry import register_trainer
from hale_vlm.training.base import Trainer


@register_trainer("vlm")
class VLMTrainer(Trainer):
    """Trainer that passes only trainable parameters to the optimizer."""

    def fit(self):
        original_build_optimizer = optim_module.build_optimizer

        def filtered_build_optimizer(opt_type, params, config):
            model = self.model
            if model is not None:
                if hasattr(model, "module"):
                    model = model.module
                if hasattr(model, "trainable_parameters"):
                    params = model.trainable_parameters()
            return original_build_optimizer(opt_type, params, config)

        optim_module.build_optimizer = filtered_build_optimizer
        try:
            return super().fit()
        finally:
            optim_module.build_optimizer = original_build_optimizer
