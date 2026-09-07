"""Download the competition data via kagglehub and place it where train.py/submit.py expect it
(`data/raw/`, matching train.py's --train-csv default of data/raw/train.csv)."""

from __future__ import annotations

import shutil
from pathlib import Path

from llm_reward.data.download import download_competition_data
from llm_reward.negative_space import require

_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def download_to_data_raw(dest_dir: Path = Path("data/raw")) -> Path:
    """Download the competition archive and copy its 3 CSVs into dest_dir. Idempotent: re-running
    overwrites whatever was there (dest_dir is git-ignored scratch, not tracked state)."""
    cache_dir = download_competition_data()
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        src = cache_dir / name
        require(src.exists(), f"expected file not found in downloaded archive: {src}")
        shutil.copy(src, dest_dir / name)
    return dest_dir


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    dest_dir = download_to_data_raw()
    print(dest_dir)


if __name__ == "__main__":
    main()
