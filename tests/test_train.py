import torch
from torch import nn

from llm_reward.models.checkpoint import Checkpoint
from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle
from llm_reward.train import (
    _best_path,
    _last_path,
    _make_optimizer,
    _make_scheduler,
    _save_checkpoint,
)


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def _lstm_config(tmp_path, **overrides) -> LSTMConfig:
    kwargs = dict(seed=1, batch_size=2, epochs=2, lr=1e-2, output_dir=tmp_path, run_name="r")
    kwargs.update(overrides)
    return LSTMConfig(**kwargs)


def test_make_optimizer_uses_flat_param_list_when_no_param_groups(tmp_path):
    model = _TinyClassifier()
    bundle = ModelBundle(model=model, collate_fn=lambda _batch: {})
    optimizer = _make_optimizer(bundle, _lstm_config(tmp_path, lr=0.05))
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.05


def test_make_optimizer_respects_bundle_param_groups(tmp_path):
    model = _TinyClassifier()
    param_groups = [
        {"params": [model.linear.weight]},
        {"params": [model.linear.bias], "lr": 0.5},
    ]
    bundle = ModelBundle(model=model, collate_fn=lambda _batch: {}, param_groups=param_groups)
    optimizer = _make_optimizer(bundle, _lstm_config(tmp_path, lr=0.01))

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 0.01  # inherits config.lr as the default
    assert optimizer.param_groups[1]["lr"] == 0.5  # keeps its own override


def test_make_scheduler_runs_without_a_model_specific_argument(tmp_path):
    model = _TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = _make_scheduler(
        optimizer, _lstm_config(tmp_path, warmup_ratio=0.1), num_training_steps=10
    )
    assert scheduler is not None


def test_last_and_best_paths_are_under_output_dir(tmp_path):
    config = _lstm_config(tmp_path)
    assert _last_path(config) == tmp_path / "last.pt"
    assert _best_path(config) == tmp_path / "best.pt"


def test_save_checkpoint_creates_parent_directories(tmp_path):
    config = _lstm_config(tmp_path / "nested" / "dir")
    checkpoint = Checkpoint(
        epoch=0, global_step=0, model_state={}, optimizer_state={}, scheduler_state=None,
        best_val_metric=0.0, config=config, wandb_run_id="r",
    )
    path = _last_path(config)
    _save_checkpoint(path, checkpoint)
    assert path.exists()
    loaded: Checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    assert loaded.epoch == 0


def test_exactly_two_checkpoint_files_exist_regardless_of_epoch_count(tmp_path):
    """Bounded disk use: only last.pt and best.pt, never one file per epoch."""
    config = _lstm_config(tmp_path)
    for epoch in range(5):
        checkpoint = Checkpoint(
            epoch=epoch, global_step=epoch * 4, model_state={}, optimizer_state={},
            scheduler_state=None, best_val_metric=float(epoch), config=config, wandb_run_id="r",
        )
        _save_checkpoint(_last_path(config), checkpoint)
        _save_checkpoint(_best_path(config), checkpoint)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt", "last.pt"]
