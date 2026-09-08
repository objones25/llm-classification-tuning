"""Download the competition data via kagglehub and place it where train.py/submit.py expect it
(`data/raw/`, matching train.py's --train-csv default of data/raw/train.csv).

By default also appends lmarena-ai/arena-human-preference-100k (converted to train.csv's own
schema) onto train.csv -- allowed under the competition's External Data rule (Section 5): the
dataset's prompts are CC-BY-4.0 and its model outputs are governed by each model's own provider
terms, the same licensing arrangement the competition's own training data is already under.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from datasets import load_dataset

from llm_reward.data.download import download_competition_data
from llm_reward.data.pairwise import TEST_COLUMNS, TRAIN_COLUMNS, require_csv_header
from llm_reward.negative_space import require
from llm_reward.submit import SUBMISSION_COLUMNS

_EXPECTED_COLUMNS = {
    "train.csv": TRAIN_COLUMNS,
    "test.csv": TEST_COLUMNS,
    "sample_submission.csv": {"id", *SUBMISSION_COLUMNS},
}

_TRAIN_COLUMN_ORDER = (
    "id", "model_a", "model_b", "prompt", "response_a", "response_b",
    "winner_model_a", "winner_model_b", "winner_tie",
)

_EXTERNAL_ARENA_DATASET = "lmarena-ai/arena-human-preference-100k"

# Verified against a real sample of the dataset (see conversation history) -- both a bare "tie"
# and a "tie (bothbad)" value occur, distinct from each other, both mapping to our one winner_tie
# column the same way train.csv's own winner_model_a/winner_model_b/winner_tie one-hot works.
_WINNER_TO_ONE_HOT = {
    "model_a": (1, 0, 0),
    "model_b": (0, 1, 0),
    "tie": (0, 0, 1),
    "tie (bothbad)": (0, 0, 1),
}


def _join_turns(conversation: list[dict], role: str) -> str:
    """Extract one side's turns for a given role as a JSON-encoded list of strings -- the exact
    on-disk encoding train.csv's own prompt/response_a/response_b columns use, so
    data/pairwise.py's _extract_text (json.loads first) handles this data through the same path
    as real Kaggle rows. Emitting already-flattened plain text here instead would make ordinary
    prose fall through to _extract_text's ast.literal_eval fallback, which emits SyntaxWarning on
    any text that merely resembles Python syntax (it isn't meant to run on prose at all)."""
    turns = [turn["content"] for turn in conversation if turn["role"] == role]
    require(len(turns) > 0, f"no {role!r} turns found in conversation")
    return json.dumps(turns)


def _convert_external_row(row: dict) -> dict[str, object]:
    """Map one lmarena-ai/arena-human-preference-100k row onto train.csv's own column schema."""
    require(row["winner"] in _WINNER_TO_ONE_HOT, f"unrecognized winner value: {row['winner']!r}")
    winner_a, winner_b, winner_tie = _WINNER_TO_ONE_HOT[row["winner"]]
    return {
        "id": row["question_id"],
        "model_a": row["model_a"],
        "model_b": row["model_b"],
        "prompt": _join_turns(row["conversation_a"], "user"),
        "response_a": _join_turns(row["conversation_a"], "assistant"),
        "response_b": _join_turns(row["conversation_b"], "assistant"),
        "winner_model_a": winner_a,
        "winner_model_b": winner_b,
        "winner_tie": winner_tie,
    }


def append_external_arena_data(train_csv: Path) -> None:
    """Download _EXTERNAL_ARENA_DATASET and append it (converted to train_csv's own schema)
    directly onto train_csv -- so load_pairwise_examples never needs to know a second data
    source exists; it just sees a longer train.csv."""
    dataset = load_dataset(_EXTERNAL_ARENA_DATASET, split="train")
    with train_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TRAIN_COLUMN_ORDER)
        for row in dataset:
            writer.writerow(_convert_external_row(row))

    require_csv_header(train_csv, TRAIN_COLUMNS)
    with train_csv.open(newline="", encoding="utf-8") as f:
        ids = [row["id"] for row in csv.DictReader(f)]
    require(len(ids) == len(set(ids)), "external data introduced duplicate ids into train.csv")


def download_to_data_raw(
    dest_dir: Path = Path("data/raw"), *, include_external: bool = True
) -> Path:
    """Download the competition archive and copy its 3 CSVs into dest_dir. Idempotent: re-running
    overwrites whatever was there (dest_dir is git-ignored scratch, not tracked state) -- always
    starting from a fresh copy of the real train.csv, so a repeated run appends external data
    exactly once rather than accumulating duplicates.

    Validates each file's header against what train.py/submit.py expect before copying -- fail
    here, at download time, rather than deep inside a training run's row-parsing loop."""
    cache_dir = download_competition_data()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_columns in _EXPECTED_COLUMNS.items():
        src = cache_dir / name
        require_csv_header(src, expected_columns)
        shutil.copy(src, dest_dir / name)

    if include_external:
        append_external_arena_data(dest_dir / "train.csv")

    return dest_dir


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Download competition (+ external) data")
    parser.add_argument(
        "--include-external",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append lmarena-ai/arena-human-preference-100k onto train.csv (default: yes)",
    )
    args = parser.parse_args()
    dest_dir = download_to_data_raw(include_external=args.include_external)
    print(dest_dir)


if __name__ == "__main__":
    main()
