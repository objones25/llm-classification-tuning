"""Pins the exact training input format.

All three variants call `format_input`, so this single assertion is what stops a future edit
from silently drifting the text one variant trains on relative to the other two.
"""

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.models._common import format_input


def test_format_input_produces_the_expected_exact_string():
    example = PairwiseExample(
        id="1", prompt="What is 2+2?", response_a="4", response_b="Four", label=0
    )
    assert format_input(example) == (
        "What is 2+2?\n[RESPONSE A]\n4\n[RESPONSE B]\nFour"
    )


def test_every_variant_shares_one_format_input_implementation():
    """The three model modules must reference the shared function, not their own copies."""
    from llm_reward.models import _hf_common, lstm_baseline

    assert lstm_baseline.format_input is format_input
    assert _hf_common.format_input is format_input
