from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Literal

import torch
import wandb
from dotenv import load_dotenv
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from .data.dataset import PairwiseDataset
from .data.pairwise import load_pairwise_examples, split_train_val
from .evaluate import evaluate
from .models.checkpoint import Checkpoint, ConfigMismatchError
from .models.config import ConfigError, TrainConfig, load_config
from .models.registry import ModelBundle, build_model
from .negative_space import require


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
    """Write via a temp file + atomic rename. A pod dying mid-`torch.save` would otherwise
    leave a truncated, unloadable `last.pt` — and `last.pt` is the only resume point there is.
    `Path.replace` is atomic on POSIX, which covers both targets (macOS dev, Linux RunPod)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def _group_grad_norm(params) -> float:
    """The L2 norm of one param group's gradients, computed WITHOUT clipping them -- purely for
    observability. Must be called before the single combined clip_grad_norm_ call below, or it
    would report the post-clip norm instead."""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm() for g in grads])).item()


def train(
    config: TrainConfig,
    bundle: ModelBundle,
    train_examples,
    val_examples,
    *,
    resume: bool,
) -> Checkpoint:
    require(config.epochs >= 1, f"epochs must be at least 1, got {config.epochs}")
    require(len(train_examples) > 0, "train() received zero training examples")
    require(len(val_examples) > 0, "train() received zero validation examples")

    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
    bundle.model.to(device)

    train_loader = DataLoader(
        PairwiseDataset(train_examples), batch_size=config.batch_size, shuffle=True,
        collate_fn=bundle.collate_fn,
    )
    val_loader = DataLoader(
        PairwiseDataset(val_examples), batch_size=config.batch_size, shuffle=False,
        collate_fn=bundle.collate_fn,
    )

    # `collate_fn`'s output shape is a static property of the bundle, so one check on one batch
    # settles it for the whole run -- the spec asks for this to fail before the training loop
    # starts, not mid-epoch. Peeking a batch here is harmless: `train_loader` shuffles, so no
    # epoch's actual iteration skips or repeats anything because of it.
    first_batch = next(iter(train_loader))
    require("labels" in first_batch, "collate_fn output is missing the required 'labels' key")

    optimizer = _make_optimizer(bundle, config)
    scheduler = _make_scheduler(optimizer, config, len(train_loader) * config.epochs)
    class_weights = (
        torch.tensor(config.class_weights, dtype=torch.float32, device=device)
        if config.class_weights else None
    )

    last_path = _last_path(config)
    run_config = dataclasses.asdict(config) | {
        "variant": config.variant,
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
    }

    checkpoint: Checkpoint | None
    wandb_run_id: str | None
    wandb_resume: Literal["must", "allow"]
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, weights_only=False, map_location=device)
        # torch.load returns Any; the assert narrows the type for the type checker
        assert isinstance(checkpoint, Checkpoint)
        # epochs is deliberately excluded from the equality check: extending a run's total epoch
        # budget across a resume is the whole point of --resume (see test_resume_continues_from_
        # the_next_epoch). Every other field still has to match exactly.
        if dataclasses.replace(checkpoint.config, epochs=config.epochs) != config:
            raise ConfigMismatchError(
                f"resume checkpoint at {last_path} was produced by a different config"
            )
        bundle.model.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        if checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)
        start_epoch = checkpoint.epoch + 1
        global_step = checkpoint.global_step
        best_val_metric = checkpoint.best_val_metric
        wandb_run_id = checkpoint.wandb_run_id
        wandb_resume = "must"
    else:
        start_epoch = 0
        global_step = 0
        best_val_metric = float("-inf")
        checkpoint = None
        wandb_run_id = None
        wandb_resume = "allow"

    with wandb.init(
        project="llm-reward", config=run_config, id=wandb_run_id, resume=wandb_resume
    ) as run:
        run.define_metric("train/global_step")
        run.define_metric("train/*", step_metric="train/global_step")
        run.define_metric("epoch")
        run.define_metric("val/*", step_metric="epoch")

        for epoch in range(start_epoch, config.epochs):
            bundle.model.train()
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_n = 0

            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                inputs = {k: v for k, v in batch.items() if k != "labels"}

                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=config.mixed_precision == "bf16"
                ):
                    logits = bundle.model(**inputs)
                    loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)

                optimizer.zero_grad()
                loss.backward()

                grad_norm_backbone: float | None = None
                grad_norm_head: float | None = None
                if bundle.param_groups is not None:
                    grad_norm_backbone = _group_grad_norm(bundle.param_groups[0]["params"])
                    grad_norm_head = _group_grad_norm(bundle.param_groups[1]["params"])
                grad_norm = nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()

                epoch_loss_sum += loss.item() * labels.shape[0]
                epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
                epoch_n += labels.shape[0]

                step_log = {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                }
                if grad_norm_backbone is not None:
                    step_log["train/grad_norm_backbone"] = grad_norm_backbone
                    step_log["train/grad_norm_head"] = grad_norm_head
                else:
                    step_log["train/grad_norm"] = float(grad_norm)
                run.log(step_log)
                global_step += 1

            metrics = evaluate(bundle.model, val_loader, device)
            epoch_log = {
                "epoch": epoch,
                "train/epoch_loss": epoch_loss_sum / epoch_n,
                "train/epoch_accuracy": epoch_correct / epoch_n,
                "val/loss": metrics.loss,
                "val/accuracy": metrics.accuracy,
                "val/macro_f1": metrics.macro_f1,
                "val/precision_a": metrics.per_class_precision[0],
                "val/precision_b": metrics.per_class_precision[1],
                "val/precision_tie": metrics.per_class_precision[2],
                "val/recall_a": metrics.per_class_recall[0],
                "val/recall_b": metrics.per_class_recall[1],
                "val/recall_tie": metrics.per_class_recall[2],
                "val/f1_a": metrics.per_class_f1[0],
                "val/f1_b": metrics.per_class_f1[1],
                "val/f1_tie": metrics.per_class_f1[2],
                "val/confusion_matrix": wandb.Table(
                    columns=["true_label", "pred_a", "pred_b", "pred_tie"],
                    data=[
                        [name, *row]
                        for name, row in zip(
                            ("a", "b", "tie"), metrics.confusion_matrix, strict=True
                        )
                    ],
                ),
            }
            # torch.accelerator has no memory-accounting API in this project's pinned torch==2.8.0
            # (added in a later release) -- go through the device-specific module instead so this
            # doesn't crash on the RunPod CUDA target or on an Apple Silicon dev machine (MPS).
            if device.type == "cuda":
                epoch_log["system/gpu_mem_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(device) / 1e6
                )
                # max_memory_allocated is cumulative since process start; reset it so the next
                # epoch's number means "peak during THIS epoch" -- otherwise the metric is a
                # monotonic high-water mark, not comparable to the MPS branch's instantaneous
                # reading under the same key.
                torch.cuda.reset_peak_memory_stats(device)
            elif device.type == "mps":
                epoch_log["system/gpu_mem_allocated_mb"] = (
                    torch.mps.current_allocated_memory() / 1e6
                )
            run.log(epoch_log)

            improved = metrics.accuracy > best_val_metric
            best_val_metric = max(best_val_metric, metrics.accuracy)
            checkpoint = Checkpoint(
                epoch=epoch,
                global_step=global_step,
                model_state=bundle.model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                best_val_metric=best_val_metric,
                config=config,
                wandb_run_id=run.id,
            )
            _save_checkpoint(last_path, checkpoint)
            if improved:
                _save_checkpoint(_best_path(config), checkpoint)

    # Reached only via the resume branch (checkpoint loaded, never None) or after the for loop
    # above ran at least once (guaranteed by the epochs >= 1 require() at the top of this
    # function), which always reassigns checkpoint to a real Checkpoint.
    assert checkpoint is not None
    return checkpoint


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train an llm-reward model variant")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        bundle = build_model(config)
        examples = load_pairwise_examples(args.train_csv)
        train_examples, val_examples = split_train_val(examples, config.val_fraction, config.seed)
        train(config, bundle, train_examples, val_examples, resume=args.resume)
    except (ConfigError, ConfigMismatchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
