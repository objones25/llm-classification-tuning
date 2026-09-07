from __future__ import annotations

import ast
import csv
import random
from dataclasses import dataclass
from pathlib import Path

from ..negative_space import bounded, require

# The real train.csv/test.csv have this asymmetric naming: only the tie column lacks the
# _model_ infix the other two winner columns carry (verified against the downloaded competition
# archive -- an earlier assumption of `winner_model_tie` crashed on the real data).
TRAIN_COLUMNS = {
    "id", "model_a", "model_b", "prompt", "response_a", "response_b",
    "winner_model_a", "winner_model_b", "winner_tie",
}
TEST_COLUMNS = {"id", "prompt", "response_a", "response_b"}


def require_csv_header(csv_path: Path, expected_columns: set[str]) -> None:
    """Fail fast with a readable message if csv_path's header doesn't match expected_columns,
    rather than letting a later row[...] access raise a bare KeyError deep in a parsing loop."""
    require(csv_path.exists(), f"csv file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = set(next(csv.reader(f), []))
    require(
        header == expected_columns,
        f"{csv_path}: expected columns {sorted(expected_columns)}, got {sorted(header)}",
    )


@dataclass(frozen=True)
class PairwiseExample:
    id: str
    prompt: str
    response_a: str
    response_b: str
    label: int  # 0=a, 1=b, 2=tie, -1=unknown (test-time, no ground truth)

    def __post_init__(self) -> None:
        require(self.label in (0, 1, 2, -1), f"label must be 0, 1, 2, or -1; got {self.label!r}")
        require(self.id, "id must not be empty")


def _extract_text(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, list):
        return "\n".join(str(turn) for turn in parsed)
    return value


def load_pairwise_examples(csv_path: Path) -> list[PairwiseExample]:
    require_csv_header(csv_path, TRAIN_COLUMNS)
    examples: list[PairwiseExample] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in bounded(reader, limit=1_000_000, name="csv rows"):
            if row["winner_model_a"] == "1":
                label = 0
            elif row["winner_model_b"] == "1":
                label = 1
            else:
                require(row["winner_tie"] == "1", f"row {row['id']} has no winner set")
                label = 2
            examples.append(
                PairwiseExample(
                    id=row["id"],
                    prompt=_extract_text(row["prompt"]),
                    response_a=_extract_text(row["response_a"]),
                    response_b=_extract_text(row["response_b"]),
                    label=label,
                )
            )
    require(len(examples) > 0, f"no rows parsed from {csv_path}")
    return examples


def load_test_examples(csv_path: Path) -> list[PairwiseExample]:
    """Load a Kaggle test.csv: same prompt/response schema as train.csv, but no winner columns
    -- there is no ground truth to leak. Every example gets label=-1 (see PairwiseExample)."""
    require_csv_header(csv_path, TEST_COLUMNS)
    examples: list[PairwiseExample] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in bounded(reader, limit=1_000_000, name="csv rows"):
            examples.append(
                PairwiseExample(
                    id=row["id"],
                    prompt=_extract_text(row["prompt"]),
                    response_a=_extract_text(row["response_a"]),
                    response_b=_extract_text(row["response_b"]),
                    label=-1,
                )
            )
    require(len(examples) > 0, f"no rows parsed from {csv_path}")
    return examples


def split_train_val(
    examples: list[PairwiseExample], val_fraction: float, seed: int
) -> tuple[list[PairwiseExample], list[PairwiseExample]]:
    require(0.0 < val_fraction < 1.0, f"val_fraction must be strictly interior, got {val_fraction}")
    require(len(examples) >= 2, f"need at least 2 examples to split, got {len(examples)}")

    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val, train = shuffled[:n_val], shuffled[n_val:]

    require(len(train) + len(val) == len(examples), "split lost or duplicated examples")
    require(
        {e.id for e in train}.isdisjoint({e.id for e in val}),
        "train and val overlap",
    )
    return train, val
