import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from llm_reward.evaluate import EvalMetrics, evaluate
from llm_reward.negative_space import CheckFailed

CPU = torch.device("cpu")


class _FixedBatchDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


class _IdentityOnFeatures(nn.Module):
    """Returns the pre-baked 'features' batch as-is. NOT nn.Identity(): its forward's parameter
    is named 'input', so calling it as model(features=...) -- the actual ModelBundle contract,
    forward(**inputs) -- raises TypeError. This class's parameter is named to match."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


class _DropoutOnFeatures(nn.Module):
    """Same fix as _IdentityOnFeatures, applied to nn.Dropout: its forward's parameter is named
    'input', so model(features=...) would raise TypeError without this wrapper."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dropout(features)


def _loader(logits_per_example: list[list[float]], labels: list[int]) -> DataLoader:
    """A DataLoader whose collate_fn hands back pre-baked logits (as 'features') and labels, so
    tests assert exact confusion-matrix-derived metrics without training anything."""

    def collate_fn(indices):
        features = torch.tensor([logits_per_example[i] for i in indices], dtype=torch.float32)
        batch_labels = torch.tensor([labels[i] for i in indices], dtype=torch.long)
        return {"features": features, "labels": batch_labels}

    return DataLoader(_FixedBatchDataset(len(labels)), batch_size=2, collate_fn=collate_fn)


def _one_hot_logit(predicted_class: int) -> list[float]:
    logit = [0.0, 0.0, 0.0]
    logit[predicted_class] = 10.0
    return logit


def test_evaluate_reports_perfect_accuracy_for_a_perfect_classifier():
    labels = [0, 1, 2, 0]
    loader = _loader([_one_hot_logit(label) for label in labels], labels)
    metrics = evaluate(_IdentityOnFeatures(), loader, CPU)
    assert isinstance(metrics, EvalMetrics)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.n_examples == 4
    assert metrics.macro_f1 == pytest.approx(1.0)


def test_evaluate_computes_confusion_matrix_and_per_class_metrics_for_imperfect_predictions():
    # true labels: 0, 0, 1, 1, 2, 2 -- predicted: 0, 1, 1, 1, 2, 0
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 0]
    loader = _loader([_one_hot_logit(p) for p in predictions], labels)

    metrics = evaluate(_IdentityOnFeatures(), loader, CPU)

    assert metrics.confusion_matrix == ((1, 1, 0), (0, 2, 0), (1, 0, 1))
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.per_class_precision == pytest.approx((0.5, 2 / 3, 1.0))
    assert metrics.per_class_recall == pytest.approx((0.5, 1.0, 0.5))
    assert metrics.per_class_f1 == pytest.approx((0.5, 0.8, 2 / 3))
    assert metrics.macro_f1 == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_evaluate_guards_against_a_class_absent_from_true_and_predicted_labels():
    # class 2 never appears as a true label or a prediction -- row/column sums are both zero.
    labels = [0, 1]
    predictions = [0, 1]
    loader = _loader([_one_hot_logit(p) for p in predictions], labels)

    metrics = evaluate(_IdentityOnFeatures(), loader, CPU)

    assert metrics.per_class_precision[2] == 0.0
    assert metrics.per_class_recall[2] == 0.0
    assert metrics.per_class_f1[2] == 0.0


def test_evaluate_rejects_an_empty_loader():
    loader = DataLoader(_FixedBatchDataset(0), batch_size=2, collate_fn=lambda x: x)
    with pytest.raises(CheckFailed, match="empty"):
        evaluate(_IdentityOnFeatures(), loader, CPU)


def test_evaluate_puts_the_model_in_eval_mode():
    model = _DropoutOnFeatures(p=0.9)  # in train() mode this would zero ~90% of activations
    loader = _loader([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [0, 1])
    evaluate(model, loader, CPU)
    assert not model.training
