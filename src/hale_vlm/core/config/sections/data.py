from typing import Literal

from hale_vlm.core.config.sections.common import StrictModel


class DataConfig(StrictModel):
    source: Literal["huggingface", "overfit"] = "huggingface"
    dataset: str = "Salesforce/wikitext"
    subset: str | None = "wikitext-2-raw-v1"
    train_split: str = "train"
    val_split: str = "validation"
    text_field: str = "text"
    cache_dir: str | None = None
    tokenizer_name: str = "gpt2"
    train_size: int | None = None
    val_size: int | None = None
    overfit_text: str | None = None
    n_overfit_copies: int = 64
    add_special_tokens: bool = False
    stride_words: int = 10
