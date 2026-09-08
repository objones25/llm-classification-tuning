"""Fetch a W&B run's metric history and save comparison plots to graphs/plots/<name>/.

train/* metrics are logged per training step (x-axis: train/global_step); val/* and the
per-epoch train/epoch_* metrics are logged once per epoch (x-axis: epoch), per
train.py's run.define_metric("val/*", step_metric="epoch") setup. To compare them on one
axis, every epoch-indexed metric here gets remapped onto the train/global_step value active
at that point in the run -- the last train/global_step logged before/at that epoch's entry.

Usage: uv run python scripts/plot_run_metrics.py <run_id> [--name lstm_baseline]
                                                            [--entity OWNER] [--project NAME]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless -- no display on RunPod or in tests
import matplotlib.pyplot as plt
import wandb

from llm_reward.negative_space import require

# Never a plottable scalar series.
_NON_METRIC_KEYS = {"_step", "_runtime", "_timestamp", "epoch", "train/global_step"}
_NON_METRIC_SUFFIXES = ("confusion_matrix",)


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def align_epoch_to_global_step(history: list[dict]) -> dict[int, int]:
    """Map each logged epoch number to the train/global_step active at that point in the run's
    history: the last train/global_step seen at or before that epoch's own log entry."""
    epoch_to_step: dict[int, int] = {}
    last_step: int | None = None
    for row in history:
        step = row.get("train/global_step")
        if not _is_missing(step):
            last_step = int(step)
        epoch = row.get("epoch")
        if not _is_missing(epoch):
            require(last_step is not None, f"epoch {epoch} logged before any train/global_step")
            epoch_to_step[int(epoch)] = last_step
    return epoch_to_step


def _step_series(history: list[dict], metric_key: str) -> tuple[list[int], list[float]]:
    """A per-step metric's (global_step, value) pairs, in logging order."""
    xs: list[int] = []
    ys: list[float] = []
    for row in history:
        value = row.get(metric_key)
        step = row.get("train/global_step")
        if not _is_missing(value) and not _is_missing(step):
            xs.append(int(step))
            ys.append(float(value))
    return xs, ys


def _epoch_series(
    history: list[dict], metric_key: str, epoch_to_step: dict[int, int]
) -> tuple[list[int], list[float]]:
    """A per-epoch metric's (global_step, value) pairs -- global_step comes from
    epoch_to_step, not the row itself, so it can share an axis with per-step metrics."""
    xs: list[int] = []
    ys: list[float] = []
    for row in history:
        value = row.get(metric_key)
        epoch = row.get("epoch")
        if not _is_missing(value) and not _is_missing(epoch):
            xs.append(epoch_to_step[int(epoch)])
            ys.append(float(value))
    return xs, ys


def _discover_metric_keys(history: list[dict]) -> tuple[set[str], set[str]]:
    """Split every plottable key seen anywhere in history into (per-step, per-epoch) sets, by
    which rows it actually appears alongside (a train/global_step row vs an epoch row)."""
    step_keys: set[str] = set()
    epoch_keys: set[str] = set()
    for row in history:
        is_step_row = not _is_missing(row.get("train/global_step"))
        is_epoch_row = not _is_missing(row.get("epoch"))
        for key, value in row.items():
            if key in _NON_METRIC_KEYS or key.endswith(_NON_METRIC_SUFFIXES):
                continue
            if _is_missing(value):
                continue
            if is_step_row:
                step_keys.add(key)
            if is_epoch_row:
                epoch_keys.add(key)
    return step_keys, epoch_keys


def _save_figure(
    output_dir: Path, name: str, series: dict[str, tuple[list[int], list[float]]], run_name: str
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (xs, ys) in series.items():
        if xs:
            ax.plot(xs, ys, label=label, marker="o", markersize=3)
    ax.set_xlabel("global_step")
    ax.set_ylabel(name)
    ax.set_title(f"{run_name} — {name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_run_metrics(history: list[dict], output_dir: Path, run_name: str) -> list[Path]:
    """Pure function: history -> saved PNG paths. No network -- unit-testable with canned data.

    run_name is a human-readable label (e.g. "lstm_baseline") embedded in every plot's title --
    distinct from the wandb run id, since a folder/title full of run ids doesn't say which
    variant produced it."""
    require(len(history) > 0, "empty history -- nothing to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_to_step = align_epoch_to_global_step(history)
    step_keys, epoch_keys = _discover_metric_keys(history)

    paths: list[Path] = []

    # Comparison plots: train vs val on the same figure, both on the global_step axis.
    loss_series = {}
    if "train/loss" in step_keys:
        loss_series["train/loss"] = _step_series(history, "train/loss")
    if "train/epoch_loss" in epoch_keys:
        loss_series["train/epoch_loss"] = _epoch_series(history, "train/epoch_loss", epoch_to_step)
    if "val/loss" in epoch_keys:
        loss_series["val/loss"] = _epoch_series(history, "val/loss", epoch_to_step)
    if loss_series:
        paths.append(_save_figure(output_dir, "loss", loss_series, run_name))

    accuracy_series = {}
    if "train/epoch_accuracy" in epoch_keys:
        accuracy_series["train/epoch_accuracy"] = _epoch_series(
            history, "train/epoch_accuracy", epoch_to_step
        )
    if "val/accuracy" in epoch_keys:
        accuracy_series["val/accuracy"] = _epoch_series(history, "val/accuracy", epoch_to_step)
    if accuracy_series:
        paths.append(_save_figure(output_dir, "accuracy", accuracy_series, run_name))

    # Standalone per-step plots.
    if "train/lr" in step_keys:
        lr_series = {"train/lr": _step_series(history, "train/lr")}
        paths.append(_save_figure(output_dir, "lr", lr_series, run_name))

    grad_norm_series = {}
    if "train/grad_norm" in step_keys:
        grad_norm_series["train/grad_norm"] = _step_series(history, "train/grad_norm")
    if "train/grad_norm_backbone" in step_keys:
        grad_norm_series["train/grad_norm_backbone"] = _step_series(
            history, "train/grad_norm_backbone"
        )
    if "train/grad_norm_head" in step_keys:
        grad_norm_series["train/grad_norm_head"] = _step_series(history, "train/grad_norm_head")
    if grad_norm_series:
        paths.append(_save_figure(output_dir, "grad_norm", grad_norm_series, run_name))

    # Standalone per-epoch val plots.
    if "val/macro_f1" in epoch_keys:
        paths.append(
            _save_figure(
                output_dir, "val_macro_f1",
                {"val/macro_f1": _epoch_series(history, "val/macro_f1", epoch_to_step)},
                run_name,
            )
        )

    for metric_group in ("precision", "recall", "f1"):
        group_series = {}
        for cls in ("a", "b", "tie"):
            key = f"val/{metric_group}_{cls}"
            if key in epoch_keys:
                group_series[key] = _epoch_series(history, key, epoch_to_step)
        if group_series:
            paths.append(_save_figure(output_dir, f"val_{metric_group}", group_series, run_name))

    if "system/gpu_mem_allocated_mb" in epoch_keys:
        paths.append(
            _save_figure(
                output_dir, "gpu_memory",
                {
                    "system/gpu_mem_allocated_mb": _epoch_series(
                        history, "system/gpu_mem_allocated_mb", epoch_to_step
                    )
                },
                run_name,
            )
        )

    require(len(paths) > 0, "no plottable metrics found in history")
    return paths


def fetch_run_history(run_id: str, *, entity: str, project: str) -> list[dict]:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.history(pandas=False, samples=1_000_000)


def plot_run(
    run_id: str, *, entity: str, project: str, output_dir: Path, name: str | None = None
) -> list[Path]:
    """name (e.g. "lstm_baseline") names both the output folder and the plot titles -- defaults
    to run_id (opaque, e.g. "0idyd8x7") when not given."""
    history = fetch_run_history(run_id, entity=entity, project=project)
    run_name = name or run_id
    return plot_run_metrics(history, output_dir / run_name, run_name)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Plot a W&B run's train/val metrics")
    parser.add_argument("run_id")
    parser.add_argument(
        "--name",
        help="Human-readable label for the output folder and plot titles "
        "(e.g. lstm_baseline) -- defaults to run_id",
    )
    parser.add_argument("--entity", default="objones25")
    parser.add_argument("--project", default="llm-reward")
    parser.add_argument("--output-dir", type=Path, default=Path("graphs/plots"))
    args = parser.parse_args()

    paths = plot_run(
        args.run_id, entity=args.entity, project=args.project, output_dir=args.output_dir,
        name=args.name,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
