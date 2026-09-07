from pathlib import Path

import pytest

from llm_reward.data.pairwise import load_pairwise_examples, load_test_examples, split_train_val
from llm_reward.negative_space import CheckFailed

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tiny_train.csv"
TEST_FIXTURE = Path(__file__).parent.parent / "fixtures" / "tiny_test.csv"


def test_load_pairwise_examples_parses_all_rows():
    examples = load_pairwise_examples(FIXTURE)
    assert len(examples) == 3
    assert [e.label for e in examples] == [0, 1, 2]


def test_load_pairwise_examples_joins_list_encoded_turns():
    examples = load_pairwise_examples(FIXTURE)
    multi_turn = next(e for e in examples if e.id == "2")
    assert "Say hi" in multi_turn.prompt
    assert "again" in multi_turn.prompt


def test_load_pairwise_examples_raises_on_missing_file(tmp_path):
    with pytest.raises(CheckFailed, match="not found"):
        load_pairwise_examples(tmp_path / "does_not_exist.csv")


def test_load_pairwise_examples_raises_on_wrong_header(tmp_path):
    bad_csv = tmp_path / "bad_train.csv"
    bad_csv.write_text("id,prompt,response_a,response_b,winner_model_a,winner_model_b\n")
    with pytest.raises(CheckFailed, match="expected columns"):
        load_pairwise_examples(bad_csv)


def test_split_train_val_is_disjoint_and_covers_everything():
    examples = load_pairwise_examples(FIXTURE) * 4  # 12 examples, still 3 unique-content rows
    examples = [
        e.__class__(
            id=str(i),
            prompt=e.prompt,
            response_a=e.response_a,
            response_b=e.response_b,
            label=e.label,
        )
        for i, e in enumerate(examples)
    ]
    train, val = split_train_val(examples, val_fraction=0.25, seed=1)
    assert len(train) + len(val) == len(examples)
    assert {e.id for e in train}.isdisjoint({e.id for e in val})


def test_load_test_examples_parses_all_rows_with_sentinel_label():
    examples = load_test_examples(TEST_FIXTURE)
    assert len(examples) == 3
    assert [e.label for e in examples] == [-1, -1, -1]
    assert [e.id for e in examples] == ["101", "102", "103"]


def test_load_test_examples_joins_list_encoded_turns():
    examples = load_test_examples(TEST_FIXTURE)
    multi_turn = next(e for e in examples if e.id == "102")
    assert "Say hi" in multi_turn.prompt
    assert "again" in multi_turn.prompt


def test_load_test_examples_raises_on_missing_file(tmp_path):
    with pytest.raises(CheckFailed, match="not found"):
        load_test_examples(tmp_path / "does_not_exist.csv")


def test_load_test_examples_raises_on_wrong_header(tmp_path):
    bad_csv = tmp_path / "bad_test.csv"
    bad_csv.write_text("id,prompt\n")
    with pytest.raises(CheckFailed, match="expected columns"):
        load_test_examples(bad_csv)


def test_split_train_val_rejects_out_of_range_fraction():
    from llm_reward.data.pairwise import PairwiseExample

    examples = [PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=0)] * 2
    with pytest.raises(CheckFailed, match="val_fraction"):
        split_train_val(examples, val_fraction=1.5, seed=1)
