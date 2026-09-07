import socket

import pytest

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.negative_space import CheckFailed


def test_tiny_pairwise_examples_covers_all_three_labels(tiny_pairwise_examples):
    labels = {example.label for example in tiny_pairwise_examples}
    assert labels == {0, 1, 2}
    assert all(isinstance(example, PairwiseExample) for example in tiny_pairwise_examples)


def test_network_is_blocked_by_default():
    with pytest.raises(CheckFailed if False else Exception, match="network connection"):
        socket.create_connection(("example.com", 80))
