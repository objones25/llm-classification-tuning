from __future__ import annotations

from dataclasses import dataclass

from ..negative_space import require


@dataclass(frozen=True)
class PairwiseExample:
    id: str
    prompt: str
    response_a: str
    response_b: str
    label: int  # 0=a, 1=b, 2=tie

    def __post_init__(self) -> None:
        require(self.label in (0, 1, 2), f"label must be 0, 1, or 2; got {self.label!r}")
        require(self.id, "id must not be empty")
