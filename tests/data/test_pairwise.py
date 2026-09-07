import pytest

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.negative_space import CheckFailed


def test_construct_valid_example():
    example = PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=0)
    assert example.label == 0
    assert example.id == "1"


@pytest.mark.parametrize("bad_label", [-1, 3, 99])
def test_rejects_out_of_range_label(bad_label):
    with pytest.raises(CheckFailed, match="label must be"):
        PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=bad_label)


def test_rejects_empty_id():
    with pytest.raises(CheckFailed, match="id must not be empty"):
        PairwiseExample(id="", prompt="p", response_a="a", response_b="b", label=0)


def test_is_frozen():
    example = PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=0)
    with pytest.raises(AttributeError):
        example.label = 1
