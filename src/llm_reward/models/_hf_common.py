"""Pieces shared by the two Hugging Face-backed variants (`sft_head.py`, `lora_head.py`).

Deliberately NOT shared with `lstm_baseline.py`: that variant hash-tokenizes to a fixed-width
`token_ids` tensor with no attention mask, so it has nothing to reuse here beyond
`_common.format_input`.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from ._common import format_input


class LogitsOnly(nn.Module):
    """Unwraps a HF ModelOutput so forward(**inputs) returns a bare logits tensor, matching the
    ModelBundle contract instead of transformers'/peft's wrapper object."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.hf_model(**inputs).logits


def make_hf_collate_fn(tokenizer, max_seq_len: int):
    def collate_fn(batch: list[PairwiseExample]) -> dict[str, torch.Tensor]:
        require(len(batch) > 0, "collate_fn received an empty batch")
        texts = [format_input(ex) for ex in batch]
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    return collate_fn
