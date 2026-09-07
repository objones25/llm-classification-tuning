from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import SFTHeadConfig
from .registry import ModelBundle, register


def _format_input(example: PairwiseExample) -> str:
    return (
        f"{example.prompt}\n[RESPONSE A]\n{example.response_a}"
        f"\n[RESPONSE B]\n{example.response_b}"
    )


class _LogitsOnly(nn.Module):
    """Unwraps a HF ModelOutput so forward(**inputs) returns a bare logits tensor, matching the
    ModelBundle contract instead of transformers' wrapper object."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.hf_model(**inputs).logits


def make_collate_fn(tokenizer, max_seq_len: int):
    def collate_fn(batch: list[PairwiseExample]) -> dict[str, torch.Tensor]:
        require(len(batch) > 0, "collate_fn received an empty batch")
        texts = [_format_input(ex) for ex in batch]
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


@register(SFTHeadConfig.variant)
def build_model(config: SFTHeadConfig) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForSequenceClassification.from_pretrained(
        config.hf_model_name, num_labels=3
    )
    hf_model.config.pad_token_id = tokenizer.pad_token_id

    if config.freeze_backbone:
        for name, param in hf_model.named_parameters():
            if not name.startswith("score"):
                param.requires_grad = False

    if config.gradient_checkpointing:
        hf_model.gradient_checkpointing_enable()
        if config.freeze_backbone:
            # Without this, gradient checkpointing recomputes the frozen embedding layer's
            # forward pass with no grad-tracking input, and backward silently produces no
            # gradient for anything downstream. Only needed when something upstream is frozen.
            hf_model.enable_input_require_grads()

    param_groups = None
    if config.head_lr is not None:
        head_params = [
            p for n, p in hf_model.named_parameters()
            if p.requires_grad and n.startswith("score")
        ]
        backbone_params = [
            p for n, p in hf_model.named_parameters()
            if p.requires_grad and not n.startswith("score")
        ]
        param_groups = [
            {"params": backbone_params},
            {"params": head_params, "lr": config.head_lr},
        ]

    return ModelBundle(
        model=_LogitsOnly(hf_model),
        collate_fn=make_collate_fn(tokenizer, config.max_seq_len),
        param_groups=param_groups,
    )
