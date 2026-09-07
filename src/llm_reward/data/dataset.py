from __future__ import annotations

from torch.utils.data import Dataset

from ..negative_space import require
from .pairwise import PairwiseExample


class PairwiseDataset(Dataset):
    def __init__(self, examples: list[PairwiseExample]) -> None:
        require(len(examples) > 0, "PairwiseDataset requires at least one example")
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> PairwiseExample:
        return self._examples[idx]
