from __future__ import annotations

import zlib

import torch
from torch import nn

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from ._common import format_input
from .config import LSTMConfig, TrainConfig
from .registry import ModelBundle, register


def _hash_tokenize(text: str, vocab_size: int, max_seq_len: int) -> list[int]:
    # zlib.crc32 (not the builtin hash()) so token ids are stable across process runs --
    # str hashing is randomized per-process (PYTHONHASHSEED / SipHash) unless the seed is
    # fixed before interpreter startup, which conftest's monkeypatch can't do retroactively.
    tokens = text.split()[:max_seq_len]
    ids = [
        zlib.crc32(tok.encode("utf-8")) % (vocab_size - 1) + 1  # 0 reserved for padding
        for tok in tokens
    ]
    ids += [0] * (max_seq_len - len(ids))
    return ids


def make_collate_fn(vocab_size: int, max_seq_len: int):
    def collate_fn(batch: list[PairwiseExample]) -> dict[str, torch.Tensor]:
        require(len(batch) > 0, "collate_fn received an empty batch")
        token_ids = torch.tensor(
            [_hash_tokenize(format_input(ex), vocab_size, max_seq_len) for ex in batch],
            dtype=torch.long,
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {"token_ids": token_ids, "labels": labels}

    return collate_fn


class LSTMClassifier(nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        # nn.LSTM's own dropout param only applies BETWEEN stacked layers, so it's a no-op at
        # num_layers=1 -- passing it unconditionally would raise nn.LSTM's own UserWarning.
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding_dropout(self.embedding(token_ids))
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(self.output_dropout(hidden[-1]))


@register(LSTMConfig.variant)
def build_model(config: TrainConfig) -> ModelBundle:
    # BuildFn takes the common TrainConfig so the registry dict stays homogeneously typed;
    # the registry only ever dispatches a variant's config to its own builder (see
    # registry.build_model), so a mismatch here is a programmer error, not an operating one.
    require(isinstance(config, LSTMConfig), f"lstm_baseline builder got a {type(config).__name__}")
    assert isinstance(config, LSTMConfig)  # redundant at runtime; narrows for the type checker
    model = LSTMClassifier(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    collate_fn = make_collate_fn(config.vocab_size, config.max_seq_len)
    return ModelBundle(model=model, collate_fn=collate_fn)
