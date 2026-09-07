"""The one definition of the training input format, shared by all three model variants.

Every variant tokenizes differently, but they must all see the SAME text for a given
`PairwiseExample` -- otherwise a comparison between variants is measuring the prompt format as
much as the model. Keeping this in one place is what makes that guarantee testable.
"""

from __future__ import annotations

from ..data.pairwise import PairwiseExample


def format_input(example: PairwiseExample) -> str:
    return (
        f"{example.prompt}\n[RESPONSE A]\n{example.response_a}"
        f"\n[RESPONSE B]\n{example.response_b}"
    )
