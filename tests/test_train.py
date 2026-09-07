import torch
from torch import nn

from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle
from llm_reward.train import _make_optimizer, _make_scheduler


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
