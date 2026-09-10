from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import TrainConfig

CollateFn = Callable[[list[PairwiseExample]], dict[str, torch.Tensor]]
StateDictFn = Callable[[nn.Module], dict]
LoadStateDictFn = Callable[[nn.Module, dict], None]


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    collate_fn: CollateFn
    param_groups: list[dict] | None = None
    # None means "save/load model.state_dict() as-is" -- the right default for a full
    # fine-tune (sft_head) or a from-scratch model (lstm_baseline). A variant whose model
    # wraps a large frozen component it shouldn't re-serialize every checkpoint (LoRA: the
    # frozen base model) supplies these to save/load only what actually needs persisting.
    state_dict_fn: StateDictFn | None = None
    load_state_dict_fn: LoadStateDictFn | None = None


BuildFn = Callable[[TrainConfig], ModelBundle]
_REGISTRY: dict[str, BuildFn] = {}


def register(variant: str) -> Callable[[BuildFn], BuildFn]:
    def _decorator(fn: BuildFn) -> BuildFn:
        require(variant not in _REGISTRY, f"variant {variant!r} already registered")
        _REGISTRY[variant] = fn
        return fn

    return _decorator


def build_model(config: TrainConfig) -> ModelBundle:
    require(
        config.variant in _REGISTRY,
        f"unknown variant {config.variant!r}; registered: {sorted(_REGISTRY)}",
    )
    return _REGISTRY[config.variant](config)


def model_state_dict(bundle: ModelBundle) -> dict:
    """The dict to persist in a Checkpoint for this bundle's model. Always call this instead of
    bundle.model.state_dict() directly, so a variant with a custom state_dict_fn (LoRA:
    adapter + head weights only, not the frozen base) is respected everywhere a checkpoint gets
    written."""
    if bundle.state_dict_fn is not None:
        return bundle.state_dict_fn(bundle.model)
    return bundle.model.state_dict()


def load_model_state_dict(bundle: ModelBundle, state_dict: dict) -> None:
    """The inverse of model_state_dict. Always call this instead of
    bundle.model.load_state_dict(...) directly -- the load path must match whichever save path
    (default or custom) actually produced the given state_dict."""
    if bundle.load_state_dict_fn is not None:
        bundle.load_state_dict_fn(bundle.model, state_dict)
    else:
        bundle.model.load_state_dict(state_dict)
