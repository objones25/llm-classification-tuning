from pathlib import Path

import pytest

from llm_reward.data.pairwise import load_pairwise_examples
from llm_reward.negative_space import CheckFailed
from scripts.download_data import _convert_external_row, download_to_data_raw, main

_TRAIN_HEADER = (
    "id,model_a,model_b,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_tie"
)
_TEST_HEADER = "id,prompt,response_a,response_b"
_SAMPLE_SUBMISSION_HEADER = "id,winner_model_a,winner_model_b,winner_tie"


def _fake_cache_dir(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "train.csv").write_text(f"{_TRAIN_HEADER}\n1,m,n,p,a,b,1,0,0\n")
    (cache_dir / "test.csv").write_text(f"{_TEST_HEADER}\n1,p,a,b\n")
    (cache_dir / "sample_submission.csv").write_text(
        f"{_SAMPLE_SUBMISSION_HEADER}\n1,0.3,0.3,0.4\n"
    )
    return cache_dir


def _external_row(question_id: str, winner: str) -> dict:
    return {
        "question_id": question_id,
        "model_a": "model-x",
        "model_b": "model-y",
        "winner": winner,
        "conversation_a": [
            {"role": "user", "content": f"prompt {question_id}"},
            {"role": "assistant", "content": f"response a {question_id}"},
        ],
        "conversation_b": [
            {"role": "user", "content": f"prompt {question_id}"},
            {"role": "assistant", "content": f"response b {question_id}"},
        ],
    }


def test_download_to_data_raw_copies_all_three_files(tmp_path, monkeypatch):
    cache_dir = _fake_cache_dir(tmp_path)
    monkeypatch.setattr(
        "scripts.download_data.download_competition_data", lambda: cache_dir
    )
    dest_dir = tmp_path / "data" / "raw"

    result = download_to_data_raw(dest_dir, include_external=False)

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

    download_to_data_raw(dest_dir, include_external=False)
    download_to_data_raw(dest_dir, include_external=False)  # must not raise on the second run

    assert (dest_dir / "train.csv").exists()


def test_download_to_data_raw_raises_on_missing_expected_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "train.csv").write_text(f"{_TRAIN_HEADER}\n1,m,n,p,a,b,1,0,0\n")
    # test.csv and sample_submission.csv deliberately missing
    monkeypatch.setattr(
        "scripts.download_data.download_competition_data", lambda: cache_dir
    )

    with pytest.raises(CheckFailed, match="not found"):
        download_to_data_raw(tmp_path / "data" / "raw", include_external=False)


def test_download_to_data_raw_raises_on_wrong_header(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "train.csv").write_text("id,prompt\n1,p\n")  # missing every other column
    (cache_dir / "test.csv").write_text(f"{_TEST_HEADER}\n1,p,a,b\n")
    (cache_dir / "sample_submission.csv").write_text(
        f"{_SAMPLE_SUBMISSION_HEADER}\n1,0.3,0.3,0.4\n"
    )
    monkeypatch.setattr(
        "scripts.download_data.download_competition_data", lambda: cache_dir
    )

    with pytest.raises(CheckFailed, match="expected columns"):
        download_to_data_raw(tmp_path / "data" / "raw", include_external=False)


def test_download_to_data_raw_appends_external_data_by_default(tmp_path, monkeypatch):
    cache_dir = _fake_cache_dir(tmp_path)
    monkeypatch.setattr("scripts.download_data.download_competition_data", lambda: cache_dir)
    fake_rows = [
        _external_row("ext-1", "model_a"),
        _external_row("ext-2", "model_b"),
        _external_row("ext-3", "tie"),
        _external_row("ext-4", "tie (bothbad)"),
    ]
    monkeypatch.setattr("scripts.download_data.load_dataset", lambda *a, **k: fake_rows)
    dest_dir = tmp_path / "data" / "raw"

    download_to_data_raw(dest_dir)  # include_external defaults to True

    lines = (dest_dir / "train.csv").read_text().splitlines()
    assert lines[0] == _TRAIN_HEADER
    assert len(lines) == 1 + 1 + len(fake_rows)  # header + original row + 4 external rows
    ids = [line.split(",")[0] for line in lines[1:]]
    assert ids == ["1", "ext-1", "ext-2", "ext-3", "ext-4"]

    # The whole point of JSON-encoding prompt/response_* (rather than pre-joining plain text) is
    # that the existing, unmodified loader parses this data through the exact same path as real
    # Kaggle rows -- prove that round trip actually works.
    examples = load_pairwise_examples(dest_dir / "train.csv")
    ext_1 = next(e for e in examples if e.id == "ext-1")
    assert ext_1.prompt == "prompt ext-1"
    assert ext_1.response_a == "response a ext-1"
    assert ext_1.response_b == "response b ext-1"


def test_download_to_data_raw_skips_external_data_when_disabled(tmp_path, monkeypatch):
    cache_dir = _fake_cache_dir(tmp_path)
    monkeypatch.setattr("scripts.download_data.download_competition_data", lambda: cache_dir)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("load_dataset should not be called when include_external=False")

    monkeypatch.setattr("scripts.download_data.load_dataset", _fail_if_called)
    dest_dir = tmp_path / "data" / "raw"

    download_to_data_raw(dest_dir, include_external=False)

    lines = (dest_dir / "train.csv").read_text().splitlines()
    assert len(lines) == 2  # header + the one original row, untouched


def test_download_to_data_raw_raises_on_duplicate_ids_from_external_data(tmp_path, monkeypatch):
    cache_dir = _fake_cache_dir(tmp_path)  # original row has id "1"
    monkeypatch.setattr("scripts.download_data.download_competition_data", lambda: cache_dir)
    monkeypatch.setattr(
        "scripts.download_data.load_dataset",
        lambda *a, **k: [_external_row("1", "model_a")],  # collides with the original id
    )

    with pytest.raises(CheckFailed, match="duplicate ids"):
        download_to_data_raw(tmp_path / "data" / "raw")


def test_convert_external_row_maps_all_winner_values_to_one_hot():
    assert _convert_external_row(_external_row("a", "model_a"))["winner_model_a"] == 1
    assert _convert_external_row(_external_row("a", "model_b"))["winner_model_b"] == 1
    assert _convert_external_row(_external_row("a", "tie"))["winner_tie"] == 1
    assert _convert_external_row(_external_row("a", "tie (bothbad)"))["winner_tie"] == 1


def test_convert_external_row_joins_multi_turn_conversations():
    row = {
        "question_id": "q1",
        "model_a": "m1",
        "model_b": "m2",
        "winner": "model_a",
        "conversation_a": [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "answer one a"},
            {"role": "user", "content": "turn two"},
            {"role": "assistant", "content": "answer two a"},
        ],
        "conversation_b": [
            {"role": "user", "content": "turn one"},
            {"role": "assistant", "content": "answer one b"},
            {"role": "user", "content": "turn two"},
            {"role": "assistant", "content": "answer two b"},
        ],
    }
    converted = _convert_external_row(row)
    # JSON-encoded lists, exactly like train.csv's own prompt/response_a/response_b columns --
    # not pre-joined plain text -- so data/pairwise.py's _extract_text (json.loads first) handles
    # this data through the same path as real Kaggle rows, never falling through to its
    # ast.literal_eval fallback (which isn't meant to run on ordinary prose at all).
    assert converted["prompt"] == '["turn one", "turn two"]'
    assert converted["response_a"] == '["answer one a", "answer two a"]'
    assert converted["response_b"] == '["answer one b", "answer two b"]'


def test_convert_external_row_raises_on_unrecognized_winner():
    with pytest.raises(CheckFailed, match="unrecognized winner"):
        _convert_external_row(_external_row("a", "nobody-won"))


def test_main_respects_no_include_external_flag(tmp_path, monkeypatch, capsys):
    cache_dir = _fake_cache_dir(tmp_path)
    monkeypatch.setattr("scripts.download_data.download_competition_data", lambda: cache_dir)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("load_dataset should not be called with --no-include-external")

    monkeypatch.setattr("scripts.download_data.load_dataset", _fail_if_called)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["download_data", "--no-include-external"])

    main()

    lines = (tmp_path / "data" / "raw" / "train.csv").read_text().splitlines()
    assert len(lines) == 2  # header + the one original row, untouched
