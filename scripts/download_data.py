"""Download the competition data via kagglehub and place it where train.py/submit.py expect it
(`data/raw/`, matching train.py's --train-csv default of data/raw/train.csv)."""

from __future__ import annotations

import shutil
from pathlib import Path

from llm_reward.data.download import download_competition_data
from llm_reward.data.pairwise import TEST_COLUMNS, TRAIN_COLUMNS, require_csv_header
from llm_reward.submit import SUBMISSION_COLUMNS

_EXPECTED_COLUMNS = {
    "train.csv": TRAIN_COLUMNS,
    "test.csv": TEST_COLUMNS,
    "sample_submission.csv": {"id", *SUBMISSION_COLUMNS},
}


def download_to_data_raw(dest_dir: Path = Path("data/raw")) -> Path:
    """Download the competition archive and copy its 3 CSVs into dest_dir. Idempotent: re-running
    overwrites whatever was there (dest_dir is git-ignored scratch, not tracked state).

    Validates each file's header against what train.py/submit.py expect before copying -- fail
    here, at download time, rather than deep inside a training run's row-parsing loop."""
    cache_dir = download_competition_data()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_columns in _EXPECTED_COLUMNS.items():
        src = cache_dir / name
        require_csv_header(src, expected_columns)
        shutil.copy(src, dest_dir / name)
    return dest_dir


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    dest_dir = download_to_data_raw()
    print(dest_dir)


if __name__ == "__main__":
    main()
