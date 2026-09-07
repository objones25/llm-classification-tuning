from pathlib import Path

import pytest

from llm_reward.negative_space import CheckFailed
from scripts.download_data import download_to_data_raw


def _fake_cache_dir(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "train.csv").write_text("id\n1\n")
    (cache_dir / "test.csv").write_text("id\n1\n")
    (cache_dir / "sample_submission.csv").write_text("id\n1\n")
    return cache_dir


def test_download_to_data_raw_copies_all_three_files(tmp_path, monkeypatch):
    cache_dir = _fake_cache_dir(tmp_path)
    monkeypatch.setattr(
        "scripts.download_data.download_competition_data", lambda: cache_dir
    )
    dest_dir = tmp_path / "data" / "raw"

    result = download_to_data_raw(dest_dir)

    assert result == dest_dir
    assert (dest_dir / "train.csv").exists()
    assert (dest_dir / "test.csv").exists()
    assert (dest_dir / "sample_submission.csv").exists()


def test_download_to_data_raw_is_idempotent(tmp_path, monkeypatch):
    cache_dir = _fake_cache_dir(tmp_path)
    monkeypatch.setattr(
        "scripts.download_data.download_competition_data", lambda: cache_dir
    )
    dest_dir = tmp_path / "data" / "raw"

    download_to_data_raw(dest_dir)
    download_to_data_raw(dest_dir)  # must not raise on the second run

    assert (dest_dir / "train.csv").exists()


def test_download_to_data_raw_raises_on_missing_expected_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "train.csv").write_text("id\n1\n")
    # test.csv and sample_submission.csv deliberately missing
    monkeypatch.setattr(
        "scripts.download_data.download_competition_data", lambda: cache_dir
    )

    with pytest.raises(CheckFailed, match="not found"):
        download_to_data_raw(tmp_path / "data" / "raw")
