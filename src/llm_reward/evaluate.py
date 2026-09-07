from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .negative_space import require


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    n_examples: int
    confusion_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    per_class_precision: tuple[float, float, float]
    per_class_recall: tuple[float, float, float]
    per_class_f1: tuple[float, float, float]
    macro_f1: float


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalMetrics:
    require(len(loader) > 0, "evaluate() received an empty DataLoader")
    model.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            logits = model(**inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_examples += labels.shape[0]
            for true_label, pred_label in zip(
                labels.tolist(), logits.argmax(dim=-1).tolist(), strict=True
            ):
                confusion[true_label][pred_label] += 1
    require(total_examples > 0, "evaluate() processed zero examples")

    correct = sum(confusion[c][c] for c in range(3))
    precision, recall, f1 = [], [], []
    for c in range(3):
        true_positive = confusion[c][c]
        predicted_count = sum(confusion[i][c] for i in range(3))
        actual_count = sum(confusion[c])
        p = true_positive / predicted_count if predicted_count > 0 else 0.0
        r = true_positive / actual_count if actual_count > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    return EvalMetrics(
        loss=total_loss / total_examples,
        accuracy=correct / total_examples,
        n_examples=total_examples,
        confusion_matrix=(tuple(confusion[0]), tuple(confusion[1]), tuple(confusion[2])),
        per_class_precision=tuple(precision),
        per_class_recall=tuple(recall),
        per_class_f1=tuple(f1),
        macro_f1=sum(f1) / 3,
    )
