import pytest
from torch.utils.data import DataLoader

from llm_reward.data.dataset import PairwiseDataset
from llm_reward.data.pairwise import PairwiseExample
from llm_reward.negative_space import CheckFailed


def test_len_matches_input_length(tiny_pairwise_examples):
    dataset = PairwiseDataset(tiny_pairwise_examples)
    assert len(dataset) == 6


def test_getitem_returns_the_raw_example_not_a_tensor(tiny_pairwise_examples):
    dataset = PairwiseDataset(tiny_pairwise_examples)
    assert dataset[0] is tiny_pairwise_examples[0]
    assert isinstance(dataset[0], PairwiseExample)


def test_rejects_empty_example_list():
    with pytest.raises(CheckFailed, match="at least one example"):
        PairwiseDataset([])


def test_works_with_a_real_dataloader_and_custom_collate_fn(tiny_pairwise_examples):
    dataset = PairwiseDataset(tiny_pairwise_examples)
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda batch: [e.id for e in batch])
    batches = list(loader)
    assert sum(len(b) for b in batches) == 6
