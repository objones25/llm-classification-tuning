from __future__ import annotations

from dataclasses import dataclass

from .config import TrainConfig


@dataclass
class Checkpoint:
    epoch: int  # last COMPLETED epoch, 0-indexed
    global_step: int  # total training steps so far, across all epochs — carried across resume so
                       # the W&B step x-axis doesn't reset to 0 on a new process
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict | None
    best_val_metric: float  # lowest val_loss seen so far (lower is better)
    epochs_without_improvement: int  # consecutive epochs since best_val_metric last improved --
                                      # carried across resume so --resume doesn't reset an
                                      # early-stopping run's patience counter back to 0
    config: TrainConfig
    wandb_run_id: str


class ConfigMismatchError(Exception):
    """Raised when a --resume checkpoint's config does not match the config just loaded from
    YAML. Operating error (someone edited the YAML between runs), not a programmer error."""
