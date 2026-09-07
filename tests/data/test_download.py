from pathlib import Path

from llm_reward.data import download


def test_download_competition_data_returns_a_path(monkeypatch):
    monkeypatch.setattr(download.kagglehub, "competition_download", lambda _: "/fake/path")
    result = download.download_competition_data()
    assert result == Path("/fake/path")


def test_download_competition_data_passes_competition_slug_through(monkeypatch):
    captured = {}

    def _fake_download(competition):
        captured["competition"] = competition
        return "/fake/path"

    monkeypatch.setattr(download.kagglehub, "competition_download", _fake_download)
    download.download_competition_data(competition="a-different-slug")
    assert captured["competition"] == "a-different-slug"
