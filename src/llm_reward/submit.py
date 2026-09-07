from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import PairwiseDataset
from .data.pairwise import load_test_examples
from .models.checkpoint import Checkpoint, ConfigMismatchError
from .models.config import ConfigError
from .models.registry import build_model
from .negative_space import require

_SUBMISSION_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")


def _read_submission_ids(sample_submission_csv: Path) -> list[str]:
    require(sample_submission_csv.exists(), f"file not found: {sample_submission_csv}")
    with sample_submission_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {"id", *_SUBMISSION_COLUMNS}
        require(
            reader.fieldnames is not None and set(reader.fieldnames) == expected,
            f"{sample_submission_csv}: expected columns id,{','.join(_SUBMISSION_COLUMNS)}, "
            f"got {reader.fieldnames}",
        )
        return [row["id"] for row in reader]


def generate_submission(
    checkpoint_path: Path,
    test_csv: Path,
    sample_submission_csv: Path,
    output_csv: Path,
) -> Path:
    """Run a trained checkpoint over test_csv and write a Kaggle-format submission.csv.

    Returns output_csv. Validates against sample_submission_csv's id set and id order rather
    than assuming test.csv already matches it -- a silently-reordered submission scores garbage
    without ever raising an error.
    """
    require(checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}")
    checkpoint: Checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    bundle = build_model(checkpoint.config)
    bundle.model.load_state_dict(checkpoint.model_state)
    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
    bundle.model.to(device)
    bundle.model.eval()

    examples = load_test_examples(test_csv)
    submission_ids = _read_submission_ids(sample_submission_csv)
    require(
        {e.id for e in examples} == set(submission_ids),
        f"{test_csv} ids do not match {sample_submission_csv} ids",
    )
    example_by_id = {e.id: e for e in examples}
    ordered_examples = [example_by_id[i] for i in submission_ids]

    loader = DataLoader(
        PairwiseDataset(ordered_examples),
        batch_size=checkpoint.config.batch_size,
        shuffle=False,
        collate_fn=bundle.collate_fn,
    )

    rows: list[tuple[str, float, float, float]] = []
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            logits = bundle.model(**inputs)
            probs = torch.softmax(logits, dim=-1).tolist()
            rows.extend(probs)

    require(len(rows) == len(ordered_examples), "prediction count does not match example count")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("id", *_SUBMISSION_COLUMNS))
        for example, prob in zip(ordered_examples, rows, strict=True):
            writer.writerow((example.id, *prob))

    return output_csv


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate a Kaggle submission.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument(
        "--sample-submission", type=Path, default=Path("data/raw/sample_submission.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    try:
        output_csv = generate_submission(
            args.checkpoint, args.test_csv, args.sample_submission, args.output
        )
    except (ConfigError, ConfigMismatchError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(output_csv)


if __name__ == "__main__":
    main()
