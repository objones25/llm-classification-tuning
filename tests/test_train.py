from typing import Any

import pytest
import torch
from torch import nn

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.evaluate import EvalMetrics
from llm_reward.models.checkpoint import Checkpoint, ConfigMismatchError
from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle
from llm_reward.negative_space import CheckFailed
from llm_reward.train import (
    _best_path,
    _last_path,
    _make_optimizer,
    _make_scheduler,
    _save_checkpoint,
    train,
)


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def _lstm_config(tmp_path, **overrides: Any) -> LSTMConfig:
    # dict[str, Any]: splatted into LSTMConfig's constructor, which has a different field type
    # per key -- a concrete value type here would make Pyright check every field against one
    # uniform type instead.
    kwargs: dict[str, Any] = dict(
        seed=1, batch_size=2, epochs=2, lr=1e-2, output_dir=tmp_path, run_name="r"
    )
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
        best_val_metric=0.0, epochs_without_improvement=0, config=config, wandb_run_id="r",
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
            scheduler_state=None, best_val_metric=float(epoch), epochs_without_improvement=0,
            config=config, wandb_run_id="r",
        )
        _save_checkpoint(_last_path(config), checkpoint)
        _save_checkpoint(_best_path(config), checkpoint)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt", "last.pt"]


class _FakeRun:
    """Stands in for the object wandb.init(...) returns. train.py holds this object and calls
    .log(...)/.id/.define_metric(...) on it, and (if used as a context manager) .__enter__/
    .__exit__ -- never module-level wandb.log(...), which targets an implicit global run."""

    def __init__(self, run_id: str, logged: list[dict]) -> None:
        self.id = run_id
        self._logged = logged

    def log(self, data: dict) -> None:
        self._logged.append(data)

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.run: _FakeRun | None = None
        self._next_id = 0

    def init(self, *, project, config=None, id=None, resume=None):
        self.run = _FakeRun(id or f"fake-run-{self._next_id}", self.logged)
        self._next_id += 1
        return self.run

    def Table(self, *, columns, data):
        return {"columns": columns, "data": data}


def _fake_bundle(with_param_groups: bool = False):
    model = _TinyClassifier()

    def collate_fn(batch: list[PairwiseExample]):
        features = torch.randn(len(batch), 4)
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {"features": features, "labels": labels}

    param_groups = None
    if with_param_groups:
        param_groups = [
            {"params": [model.linear.weight]},
            {"params": [model.linear.bias], "lr": 0.5},
        ]
    return ModelBundle(model=model, collate_fn=collate_fn, param_groups=param_groups)


def _examples(n: int) -> list[PairwiseExample]:
    return [
        PairwiseExample(id=str(i), prompt="p", response_a="a", response_b="b", label=i % 3)
        for i in range(n)
    ]


def _epoch_entries(logged: list[dict]) -> list[dict]:
    """Per-epoch summary entries (they carry an 'epoch' key); per-step entries don't."""
    return [entry for entry in logged if "epoch" in entry]


def _canned_metrics(losses: list[float]):
    """Stands in for evaluate(): returns one EvalMetrics per call, loss taken from `losses` in
    order, so a test can script an exact val_loss curve instead of depending on what a tiny
    randomly-initialized model happens to produce."""
    it = iter(losses)

    def fake_evaluate(model, loader, device):
        return EvalMetrics(
            loss=next(it),
            accuracy=0.5,
            n_examples=1,
            confusion_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)),
            per_class_precision=(0.0, 0.0, 0.0),
            per_class_recall=(0.0, 0.0, 0.0),
            per_class_f1=(0.0, 0.0, 0.0),
            macro_f1=0.0,
        )

    return fake_evaluate


def test_train_runs_and_writes_both_checkpoint_files(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=2, batch_size=2)
    bundle = _fake_bundle()

    train(config, bundle, _examples(8), _examples(4), resume=False)

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_train_moves_model_and_batches_to_the_resolved_device(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)
    bundle = _fake_bundle()

    train(config, bundle, _examples(8), _examples(4), resume=False)

    # Compare against the same resolution logic train() uses, rather than hardcoding "cpu": this
    # asserts the model actually landed on whatever device train() resolved (CPU in CI with no
    # GPU; MPS on an Apple Silicon dev machine; CUDA on the RunPod pod), not merely that training
    # didn't crash.
    expected_device = torch.accelerator.current_accelerator(check_available=True) or torch.device(
        "cpu"
    )
    assert next(bundle.model.parameters()).device.type == expected_device.type


def test_train_logs_per_step_loss_lr_and_grad_norm(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    step_entries = [entry for entry in fake_wandb.logged if "train/global_step" in entry]
    assert len(step_entries) == 4  # 8 train examples / batch_size 2
    assert {
        "train/loss", "train/lr", "train/grad_norm", "train/global_step",
    } <= step_entries[0].keys()
    assert [entry["train/global_step"] for entry in step_entries] == [0, 1, 2, 3]


def test_train_logs_split_grad_norm_when_bundle_has_param_groups(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(with_param_groups=True), _examples(8), _examples(4), resume=False)

    step_entries = [entry for entry in fake_wandb.logged if "train/global_step" in entry]
    assert "train/grad_norm_backbone" in step_entries[0]
    assert "train/grad_norm_head" in step_entries[0]
    assert "train/grad_norm" not in step_entries[0]


def test_train_logs_epoch_summary_with_full_metric_set(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=2, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    epoch_entries = _epoch_entries(fake_wandb.logged)
    assert [entry["epoch"] for entry in epoch_entries] == [0, 1]
    assert {
        "train/epoch_loss", "train/epoch_accuracy", "val/loss", "val/accuracy", "val/macro_f1",
        "val/precision_a", "val/recall_a", "val/f1_a", "val/confusion_matrix",
    } <= epoch_entries[0].keys()


def test_resume_continues_from_the_next_epoch(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=2, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)
    assert [entry["epoch"] for entry in _epoch_entries(fake_wandb.logged)] == [0, 1]

    resumed_config = _lstm_config(tmp_path, epochs=4, batch_size=2)
    fake_wandb.logged.clear()
    train(resumed_config, _fake_bundle(), _examples(8), _examples(4), resume=True)
    assert [entry["epoch"] for entry in _epoch_entries(fake_wandb.logged)] == [2, 3]


def test_resume_continues_the_global_step_counter(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)  # 4 steps: 0..3

    resumed_config = _lstm_config(tmp_path, epochs=2, batch_size=2)
    fake_wandb.logged.clear()
    train(resumed_config, _fake_bundle(), _examples(8), _examples(4), resume=True)
    step_entries = [entry for entry in fake_wandb.logged if "train/global_step" in entry]
    assert [entry["train/global_step"] for entry in step_entries] == [4, 5, 6, 7]


def test_resume_with_a_wandb_run_id_reuses_it(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)
    assert fake_wandb.run is not None  # train() always calls wandb.init(), which sets it
    first_run_id = fake_wandb.run.id

    resumed_config = _lstm_config(tmp_path, epochs=2, batch_size=2)
    train(resumed_config, _fake_bundle(), _examples(8), _examples(4), resume=True)
    assert fake_wandb.run is not None
    assert fake_wandb.run.id == first_run_id


def test_resume_with_a_changed_config_raises_config_mismatch_error(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)
    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    different_config = _lstm_config(tmp_path, epochs=2, batch_size=4)  # batch_size changed
    with pytest.raises(ConfigMismatchError):
        train(different_config, _fake_bundle(), _examples(8), _examples(4), resume=True)


def test_best_checkpoint_criterion_is_val_loss_not_accuracy(tmp_path, monkeypatch):
    """Accuracy improves every epoch here while loss gets worse after epoch 0 -- best.pt must
    still land on epoch 0, proving the criterion really is loss, not accuracy."""
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    monkeypatch.setattr(train_module, "evaluate", _canned_metrics([1.0, 1.5, 1.6]))
    config = _lstm_config(tmp_path, epochs=3, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    best: Checkpoint = torch.load(_best_path(config), weights_only=False, map_location="cpu")
    assert best.epoch == 0
    assert best.best_val_metric == 1.0


def test_early_stopping_disabled_by_default_runs_every_epoch(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    monkeypatch.setattr(train_module, "evaluate", _canned_metrics([1.0, 1.1, 1.2, 1.3]))
    config = _lstm_config(tmp_path, epochs=4, batch_size=2)  # early_stopping_patience=None

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    assert [entry["epoch"] for entry in _epoch_entries(fake_wandb.logged)] == [0, 1, 2, 3]


def test_early_stopping_breaks_after_patience_epochs_without_improvement(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    # loss improves at epoch 0, then never again -- patience=2 should stop after epoch 2
    # (epochs 1 and 2 both fail to improve on epoch 0's 1.0).
    monkeypatch.setattr(train_module, "evaluate", _canned_metrics([1.0, 1.1, 1.2, 1.3, 1.4]))
    config = _lstm_config(tmp_path, epochs=10, batch_size=2, early_stopping_patience=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    assert [entry["epoch"] for entry in _epoch_entries(fake_wandb.logged)] == [0, 1, 2]


def test_resume_carries_over_epochs_without_improvement(tmp_path, monkeypatch):
    """A resumed run's patience counter must continue where it left off, not reset to 0 -- else
    --resume would silently grant extra patience an uninterrupted run would never have had."""
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    monkeypatch.setattr(train_module, "evaluate", _canned_metrics([1.0, 1.1]))
    config = _lstm_config(tmp_path, epochs=2, batch_size=2, early_stopping_patience=3)
    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    last: Checkpoint = torch.load(_last_path(config), weights_only=False, map_location="cpu")
    assert last.epochs_without_improvement == 1

    fake_wandb.logged.clear()
    monkeypatch.setattr(train_module, "evaluate", _canned_metrics([1.2, 1.3]))
    resumed_config = _lstm_config(
        tmp_path, epochs=4, batch_size=2, early_stopping_patience=3
    )
    train(resumed_config, _fake_bundle(), _examples(8), _examples(4), resume=True)

    # epoch 2 -> epochs_without_improvement 2, epoch 3 -> 3 == patience -> stops there
    assert [entry["epoch"] for entry in _epoch_entries(fake_wandb.logged)] == [2, 3]


def test_train_rejects_zero_epochs(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=0, batch_size=2)
    with pytest.raises(CheckFailed, match="epochs"):
        train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)


def test_train_rejects_non_positive_early_stopping_patience(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=1, batch_size=2, early_stopping_patience=0)
    with pytest.raises(CheckFailed, match="early_stopping_patience"):
        train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)


def test_train_rejects_empty_train_examples(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)
    with pytest.raises(CheckFailed, match="training examples"):
        train(config, _fake_bundle(), [], _examples(4), resume=False)


def test_train_rejects_a_collate_fn_that_omits_labels_before_the_loop_starts(
    tmp_path, monkeypatch
):
    """collate_fn's output shape is static, so this is checked once on the first batch rather
    than once per step -- and it must fail before any W&B run is opened, not mid-epoch."""
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    bundle = ModelBundle(
        model=_TinyClassifier(),
        collate_fn=lambda batch: {"features": torch.randn(len(batch), 4)},
    )
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    with pytest.raises(CheckFailed, match="labels"):
        train(config, bundle, _examples(8), _examples(4), resume=False)

    assert fake_wandb.run is None  # failed fast: never got as far as starting a run
