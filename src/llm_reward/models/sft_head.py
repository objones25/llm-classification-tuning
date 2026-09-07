from __future__ import annotations

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..negative_space import require
from ._hf_common import LogitsOnly, make_hf_collate_fn
from .config import SFTHeadConfig, TrainConfig
from .registry import ModelBundle, register


@register(SFTHeadConfig.variant)
def build_model(config: TrainConfig) -> ModelBundle:
    # BuildFn takes the common TrainConfig so the registry dict stays homogeneously typed;
    # the registry only ever dispatches a variant's config to its own builder (see
    # registry.build_model), so a mismatch here is a programmer error, not an operating one.
    require(
        isinstance(config, SFTHeadConfig), f"small_sft_head builder got a {type(config).__name__}"
    )
    assert isinstance(config, SFTHeadConfig)  # redundant at runtime; narrows for the type checker
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
        require(
            len(head_params) > 0,
            "no head parameters found -- check the model architecture's classification head naming",
        )
        backbone_params = [
            p for n, p in hf_model.named_parameters()
            if p.requires_grad and not n.startswith("score")
        ]
        param_groups = [
            {"params": backbone_params},
            {"params": head_params, "lr": config.head_lr},
        ]

    return ModelBundle(
        model=LogitsOnly(hf_model),
        collate_fn=make_hf_collate_fn(tokenizer, config.max_seq_len),
        param_groups=param_groups,
    )
