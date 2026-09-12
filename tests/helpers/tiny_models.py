"""Tiny stand-ins for HuggingFace models used in integration tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class _TinyConfig:
    hidden_size: int = 32
    vocab_size: int = 128
    model_type: str = "tiny"


class TinyTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    unk_token_id = 1
    vocab_size = 128

    def __len__(self) -> int:
        return self.vocab_size

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<image>":
            return 3
        return 4

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        return 0

    def __call__(
        self,
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        ids = [3] + [4 + (ord(c) % 100) for c in text]
        if truncation:
            ids = ids[:max_length]
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        attention = [1 if i != self.pad_token_id else 0 for i in ids]
        tensor = torch.tensor(ids, dtype=torch.long)
        mask = torch.tensor(attention, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": tensor.unsqueeze(0), "attention_mask": mask.unsqueeze(0)}
        return {"input_ids": ids, "attention_mask": attention}

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(chr(32 + (int(i) % 95)) for i in ids)


class TinyVisionOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class TinyVisionModel(nn.Module):
    config_class = _TinyConfig

    def __init__(self, config: _TinyConfig | None = None) -> None:
        super().__init__()
        self.config = config or _TinyConfig()
        self.proj = nn.Linear(3, self.config.hidden_size)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def forward(self, pixel_values: torch.Tensor):
        batch = pixel_values.shape[0]
        tokens = pixel_values.mean(dim=(2, 3)).unsqueeze(1).expand(batch, 16, 3)
        return TinyVisionOutput(self.proj(tokens))


class TinyCausalLM(nn.Module):
    config_class = _TinyConfig

    def __init__(self, config: _TinyConfig | None = None) -> None:
        super().__init__()
        self.config = config or _TinyConfig()
        self.embed_tokens_layer = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.q_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def get_input_embeddings(self):
        return self.embed_tokens_layer

    def set_input_embeddings(self, value):
        self.embed_tokens_layer = value

    def resize_token_embeddings(self, new_size: int):
        old = self.embed_tokens_layer
        new = nn.Embedding(new_size, self.config.hidden_size)
        new.weight.data[: old.num_embeddings] = old.weight.data
        self.embed_tokens_layer = new
        self.config.vocab_size = new_size
        return new

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **_kwargs,
    ):
        if inputs_embeds is None:
            x = self.embed_tokens_layer(input_ids)
        else:
            x = inputs_embeds
        x = self.o_proj(self.v_proj(x))
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return type("LMOutput", (), {"loss": loss, "logits": logits})()

    def generate(self, inputs_embeds, attention_mask=None, max_new_tokens=16, **_kwargs):
        batch, seq_len, hidden = inputs_embeds.shape
        generated = []
        state = inputs_embeds
        for _ in range(max_new_tokens):
            logits = self.lm_head(self.o_proj(self.v_proj(state)))
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            next_embed = self.embed_tokens_layer(next_token)
            state = torch.cat([state, next_embed], dim=1)
        prefix = torch.zeros(batch, seq_len, dtype=torch.long)
        return torch.cat([prefix, torch.cat(generated, dim=1)], dim=1)
