from pathlib import Path

import pytest

from llm_reward.data import download


def test_download_competition_data_returns_a_path(monkeypatch):
    monkeypatch.setattr(download.kagglehub, "competition_download", lambda _: "/fake/path")
    result = download.download_competition_data()
    assert result == Path("/fake/path")


def test_download_competition_data_does_not_retry_when_the_first_call_succeeds(monkeypatch):
    calls = []

    def _fake_download(competition):
        calls.append(competition)
        return "/fake/path"

    monkeypatch.setattr(download.kagglehub, "competition_download", _fake_download)
    assert download.download_competition_data() == Path("/fake/path")
    assert len(calls) == 1


def test_download_competition_data_retries_max_retries_times_then_raises(monkeypatch):
    calls = []

    def _always_fails(competition):
        calls.append(competition)
        raise ConnectionError("kaggle is down")

    monkeypatch.setattr(download.kagglehub, "competition_download", _always_fails)
    with pytest.raises(RuntimeError, match="failed after 3 attempts") as exc_info:
        download.download_competition_data(max_retries=3)

    assert len(calls) == 3
    assert isinstance(exc_info.value.__cause__, ConnectionError)


def test_download_competition_data_passes_competition_slug_through(monkeypatch):
    captured = {}

    def _fake_download(competition):
        captured["competition"] = competition
        return "/fake/path"

    monkeypatch.setattr(download.kagglehub, "competition_download", _fake_download)
    download.download_competition_data(competition="a-different-slug")
    assert captured["competition"] == "a-different-slug"
