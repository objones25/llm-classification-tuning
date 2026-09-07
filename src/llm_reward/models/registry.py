from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import TrainConfig

CollateFn = Callable[[list[PairwiseExample]], dict[str, torch.Tensor]]


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    collate_fn: CollateFn
    param_groups: list[dict] | None = None


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
