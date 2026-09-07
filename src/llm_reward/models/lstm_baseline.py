from __future__ import annotations

import torch
from torch import nn

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import LSTMConfig
from .registry import ModelBundle, register


def _format_input(example: PairwiseExample) -> str:
    return (
        f"{example.prompt}\n[RESPONSE A]\n{example.response_a}"
        f"\n[RESPONSE B]\n{example.response_b}"
    )


def _hash_tokenize(text: str, vocab_size: int, max_seq_len: int) -> list[int]:
    tokens = text.split()[:max_seq_len]
    ids = [hash(tok) % (vocab_size - 1) + 1 for tok in tokens]  # 0 reserved for padding
    ids += [0] * (max_seq_len - len(ids))
    return ids


def make_collate_fn(vocab_size: int, max_seq_len: int):
    def collate_fn(batch: list[PairwiseExample]) -> dict[str, torch.Tensor]:
        require(len(batch) > 0, "collate_fn received an empty batch")
        token_ids = torch.tensor(
            [_hash_tokenize(_format_input(ex), vocab_size, max_seq_len) for ex in batch],
            dtype=torch.long,
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {"token_ids": token_ids, "labels": labels}

    return collate_fn


class LSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden[-1])


@register(LSTMConfig.variant)
def build_model(config: LSTMConfig) -> ModelBundle:
    model = LSTMClassifier(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    collate_fn = make_collate_fn(config.vocab_size, config.max_seq_len)
    return ModelBundle(model=model, collate_fn=collate_fn)
