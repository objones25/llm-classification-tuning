from __future__ import annotations

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..negative_space import require
from ._hf_common import LogitsOnly, make_hf_collate_fn
from .config import LoRAConfig, TrainConfig
from .registry import ModelBundle, register


@register(LoRAConfig.variant)
def build_model(config: TrainConfig) -> ModelBundle:
    # BuildFn takes the common TrainConfig so the registry dict stays homogeneously typed;
    # the registry only ever dispatches a variant's config to its own builder (see
    # registry.build_model), so a mismatch here is a programmer error, not an operating one.
    require(isinstance(config, LoRAConfig), f"medium_lora builder got a {type(config).__name__}")
    assert isinstance(config, LoRAConfig)  # redundant at runtime; narrows for the type checker
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
        # Both methods are real PeftModel attributes, forwarded at runtime via __getattr__ to the
        # wrapped base model. peft's __getattr__ has no return annotation, so Pyright falls back to
        # nn.Module.__getattr__'s stubbed `Tensor | Module` return type and misreads them as
        # non-callable Tensors -- a stub gap, not a real type error.
        peft_model.gradient_checkpointing_enable()  # pyright: ignore[reportCallIssue]
        # LoRA always freezes the base, so this is always needed here (unlike sft_head.py, where
        # it's conditional on freeze_backbone) — without it, gradient checkpointing recomputes the
        # frozen embedding layer's forward pass with no grad-tracking input, and backward silently
        # produces no gradient for the adapters at all.
        peft_model.enable_input_require_grads()  # pyright: ignore[reportCallIssue]

    param_groups = None
    if config.head_lr is not None:
        head_params = [
            p
            for n, p in peft_model.named_parameters()
            if p.requires_grad and "modules_to_save" in n
        ]
        require(
            len(head_params) > 0,
            "no head parameters found -- check the model architecture's classification head naming",
        )
        adapter_params = [
            p
            for n, p in peft_model.named_parameters()
            if p.requires_grad and "modules_to_save" not in n
        ]
        param_groups = [{"params": adapter_params}, {"params": head_params, "lr": config.head_lr}]

    return ModelBundle(
        model=LogitsOnly(peft_model),
        collate_fn=make_hf_collate_fn(tokenizer, config.max_seq_len),
        param_groups=param_groups,
    )
