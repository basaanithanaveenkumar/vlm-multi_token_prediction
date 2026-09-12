from typing import Literal

from hale_vlm.core.config.sections.common import StrictModel


class ModelConfig(StrictModel):
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    max_length: int = 64
    attn_type: Literal["causal", "bidirectional", "block_causal"] = "causal"
    attn_impl: Literal["mha", "gqa", "mqa"] = "mha"
    n_kv_heads: int | None = None
    ffn_type: Literal["mlp", "geglu", "moe"] = "mlp"
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_num_shared: int = 1
    use_time_cond: bool = False
    block_size: int | None = None
    n_mtp_heads: int = 2
    arch: Literal["default", "transformer", "lgt", "dit"] = "default"
    sliding_window: int = 512
    local_global_ratio: int = 5
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    p_rope: float = 0.25
    qk_norm: bool = False
