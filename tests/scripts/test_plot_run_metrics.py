
import pytest

from llm_reward.negative_space import CheckFailed
from scripts.plot_run_metrics import (
    _save_figure,
    _step_series,
    align_epoch_to_global_step,
    plot_run_metrics,
)

NAN = float("nan")


def _lstm_style_history() -> list[dict]:
    """2 steps per epoch, 2 epochs -- plain train/grad_norm (no param_groups)."""
    rows = []
    step = 0
    for epoch in range(2):
        for _ in range(2):
            rows.append({
                "train/loss": 1.0 - 0.1 * step,
                "train/lr": 0.01,
                "train/grad_norm": 0.5,
                "train/global_step": step,
                "epoch": NAN,
                "val/loss": NAN,
                "val/accuracy": NAN,
            })
            step += 1
        rows.append({
            "epoch": epoch,
            "train/epoch_loss": 0.9 - 0.1 * epoch,
            "train/epoch_accuracy": 0.4 + 0.1 * epoch,
            "val/loss": 1.1 - 0.05 * epoch,
            "val/accuracy": 0.35 + 0.02 * epoch,
            "val/macro_f1": 0.3 + 0.01 * epoch,
            "val/precision_a": 0.4,
            "val/precision_b": 0.4,
            "val/precision_tie": 0.3,
            "val/recall_a": 0.4,
            "val/recall_b": 0.4,
            "val/recall_tie": 0.3,
            "val/f1_a": 0.4,
            "val/f1_b": 0.4,
            "val/f1_tie": 0.3,
            "system/gpu_mem_allocated_mb": 500.0 + epoch,
            "train/global_step": NAN,
            "train/loss": NAN,
        })
    return rows


def test_align_epoch_to_global_step_uses_last_step_before_each_epoch():
    history = _lstm_style_history()
    mapping = align_epoch_to_global_step(history)
    assert mapping == {0: 1, 1: 3}


def test_align_epoch_to_global_step_raises_if_epoch_logged_before_any_step():
    history = [{"epoch": 0, "val/loss": 1.0}]
    with pytest.raises(CheckFailed, match="logged before any train/global_step"):
        align_epoch_to_global_step(history)


def test_plot_run_metrics_writes_expected_files_for_lstm_style_run(tmp_path):
    paths = plot_run_metrics(_lstm_style_history(), tmp_path, "lstm_baseline")

    names = {p.name for p in paths}
    assert names == {
        "loss.png", "accuracy.png", "lr.png", "grad_norm.png",
        "val_macro_f1.png", "val_precision.png", "val_recall.png", "val_f1.png",
        "gpu_memory.png",
    }
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def _sft_head_style_history() -> list[dict]:
    """param_groups set -- grad_norm_backbone/grad_norm_head instead of plain grad_norm."""
    return [
        {
            "train/loss": 1.0, "train/lr": 0.001,
            "train/grad_norm_backbone": 2.0, "train/grad_norm_head": 0.1,
            "train/global_step": 0, "epoch": NAN,
        },
        {
            "epoch": 0, "train/epoch_loss": 0.9, "train/epoch_accuracy": 0.5,
            "val/loss": 1.0, "val/accuracy": 0.4, "val/macro_f1": 0.35,
            "val/precision_a": 0.4, "val/precision_b": 0.4, "val/precision_tie": 0.3,
            "val/recall_a": 0.4, "val/recall_b": 0.4, "val/recall_tie": 0.3,
            "val/f1_a": 0.4, "val/f1_b": 0.4, "val/f1_tie": 0.3,
            "system/gpu_mem_allocated_mb": 29000.0,
            "train/global_step": NAN,
        },
    ]


def test_plot_run_metrics_uses_split_grad_norm_when_present(tmp_path):
    paths = plot_run_metrics(_sft_head_style_history(), tmp_path, "small_sft_head")
    assert "grad_norm.png" in {p.name for p in paths}

    # The plot's filename is the same either way ("grad_norm.png") -- what actually differs is
    # which series feed it. Confirm the split keys extract real data from this history.
    backbone_xs, backbone_ys = _step_series(_sft_head_style_history(), "train/grad_norm_backbone")
    head_xs, head_ys = _step_series(_sft_head_style_history(), "train/grad_norm_head")
    assert backbone_xs == [0]
    assert backbone_ys == [2.0]
    assert head_xs == [0]
    assert head_ys == [0.1]


def test_plot_run_metrics_raises_on_empty_history(tmp_path):
    with pytest.raises(CheckFailed, match="empty history"):
        plot_run_metrics([], tmp_path, "lstm_baseline")


def test_plot_run_metrics_creates_output_directory(tmp_path):
    dest = tmp_path / "nested" / "run-id"
    plot_run_metrics(_lstm_style_history(), dest, "lstm_baseline")
    assert dest.exists()


def test_save_figure_embeds_run_name_in_the_title(tmp_path):
    fig_path = _save_figure(tmp_path, "loss", {"train/loss": ([0, 1], [1.0, 0.5])}, "lstm_baseline")
    assert fig_path.exists()


def test_plot_run_titles_include_the_run_name(tmp_path, monkeypatch):
    """_save_figure closes its figure before returning, so assert on the title through a spy
    instead of trying to inspect a closed Figure."""
    import scripts.plot_run_metrics as plot_module

    seen_titles: list[str] = []
    real_save_figure = plot_module._save_figure

    def _spy(output_dir, name, series, run_name):
        seen_titles.append(f"{run_name} — {name}")
        return real_save_figure(output_dir, name, series, run_name)

    monkeypatch.setattr(plot_module, "_save_figure", _spy)
    plot_module.plot_run_metrics(_lstm_style_history(), tmp_path, "lstm_baseline")

    assert "lstm_baseline — loss" in seen_titles
    assert "lstm_baseline — accuracy" in seen_titles
