from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import get_scheduler

from .models.checkpoint import Checkpoint
from .models.config import TrainConfig
from .models.registry import ModelBundle


def _make_optimizer(bundle: ModelBundle, config: TrainConfig) -> AdamW:
    params = (
        bundle.param_groups
        if bundle.param_groups is not None
        else [p for p in bundle.model.parameters() if p.requires_grad]
    )
    return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def _make_scheduler(optimizer: AdamW, config: TrainConfig, num_training_steps: int):
    return get_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(config.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )


def _last_path(config: TrainConfig) -> Path:
    return Path(config.output_dir) / "last.pt"


def _best_path(config: TrainConfig) -> Path:
    return Path(config.output_dir) / "best.pt"


def _save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
