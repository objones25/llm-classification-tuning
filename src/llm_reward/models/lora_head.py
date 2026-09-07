from __future__ import annotations

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import LoRAConfig
from .registry import ModelBundle, register


def _format_input(example: PairwiseExample) -> str:
    return (
        f"{example.prompt}\n[RESPONSE A]\n{example.response_a}\n"
        f"[RESPONSE B]\n{example.response_b}"
    )


class _LogitsOnly(nn.Module):
    """Unwraps a HF ModelOutput so forward(**inputs) returns a bare logits tensor, matching the
    ModelBundle contract instead of transformers'/peft's wrapper object. Duplicated from
    sft_head.py rather than shared, per the spec: registry.py stays stable, this file stays
    independent of the sft_head track."""

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


@register(LoRAConfig.variant)
def build_model(config: LoRAConfig) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.hf_model_name, num_labels=3
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = PeftLoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        task_type="SEQ_CLS",
    )
    peft_model = get_peft_model(base_model, lora_config)

    if config.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()
        # LoRA always freezes the base, so this is always needed here (unlike sft_head.py, where
        # it's conditional on freeze_backbone) — without it, gradient checkpointing recomputes the
        # frozen embedding layer's forward pass with no grad-tracking input, and backward silently
        # produces no gradient for the adapters at all.
        peft_model.enable_input_require_grads()

    param_groups = None
    if config.head_lr is not None:
        head_params = [
            p
            for n, p in peft_model.named_parameters()
            if p.requires_grad and "modules_to_save" in n
        ]
        adapter_params = [
            p
            for n, p in peft_model.named_parameters()
            if p.requires_grad and "modules_to_save" not in n
        ]
        param_groups = [{"params": adapter_params}, {"params": head_params, "lr": config.head_lr}]

    return ModelBundle(
        model=_LogitsOnly(peft_model),
        collate_fn=make_collate_fn(tokenizer, config.max_seq_len),
        param_groups=param_groups,
    )
