from pathlib import Path

import pytest

from llm_reward.data.pairwise import (
    _extract_text,
    load_pairwise_examples,
    load_test_examples,
    split_train_val,
)
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


def test_load_pairwise_examples_parses_a_field_larger_than_csvs_default_limit(tmp_path):
    """csv's stdlib default field_size_limit is 131072 bytes -- too small for a real multi-turn
    LLM conversation joined into one field. This must not raise _csv.Error: field larger than
    field limit."""
    big_csv = tmp_path / "big_field.csv"
    huge_prompt = "x" * 200_000  # bigger than csv's 131072-byte default limit
    with big_csv.open("w", newline="", encoding="utf-8") as f:
        f.write(
            "id,model_a,model_b,prompt,response_a,response_b,"
            "winner_model_a,winner_model_b,winner_tie\n"
        )
        f.write(f'1,m,n,"{huge_prompt}",a,b,1,0,0\n')

    examples = load_pairwise_examples(big_csv)
    assert len(examples[0].prompt) == 200_000


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


def test_extract_text_joins_a_real_style_json_double_quoted_list(recwarn):
    result = _extract_text('["Say hi", "again"]')
    assert result == "Say hi\nagain"
    assert len(recwarn) == 0  # ast.literal_eval-first would warn on nothing here, but pin it


def test_extract_text_handles_json_escaped_forward_slash_without_warning(recwarn):
    # A real value from the competition data: JSON's \/ escape for a literal /. ast.literal_eval
    # doesn't recognize \/ as an escape sequence and emits SyntaxWarning -- once per row, which
    # crashed nothing but flooded stdout on a real training run.
    result = _extract_text(r'["a slash: \/ here"]')
    assert result == "a slash: / here"
    assert len(recwarn) == 0


def test_extract_text_falls_back_to_ast_literal_eval_for_single_quoted_list():
    result = _extract_text("['Say hi', 'again']")
    assert result == "Say hi\nagain"


def test_extract_text_returns_raw_string_for_plain_non_list_text():
    assert _extract_text("What is 2+2?") == "What is 2+2?"


def test_extract_text_returns_raw_string_when_it_looks_like_a_set_of_dicts():
    # Real bug: text starting with "{" that isn't valid JSON parses under ast.literal_eval as a
    # set literal containing dict elements -- constructing that set raises TypeError (dicts are
    # unhashable), not ValueError/SyntaxError, which _extract_text's fallback used to miss.
    value = '{{"role": "user"}, {"role": "assistant"}}'
    assert _extract_text(value) == value


def test_split_train_val_rejects_out_of_range_fraction():
    from llm_reward.data.pairwise import PairwiseExample

    examples = [PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=0)] * 2
    with pytest.raises(CheckFailed, match="val_fraction"):
        split_train_val(examples, val_fraction=1.5, seed=1)
