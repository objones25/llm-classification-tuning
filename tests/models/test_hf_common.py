import pytest
import torch

from llm_reward.models._common import format_input
from llm_reward.models._hf_common import SYSTEM_MESSAGE, make_hf_collate_fn
from llm_reward.negative_space import CheckFailed


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.last_call: dict | None = None

    def apply_chat_template(
        self, conversations, *, tokenize, add_generation_prompt, padding, truncation,
        max_length, return_tensors, return_dict,
    ):
        self.last_call = {
            "conversations": conversations,
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "padding": padding,
            "truncation": truncation,
            "max_length": max_length,
            "return_tensors": return_tensors,
            "return_dict": return_dict,
        }
        batch_size = len(conversations)
        seq_len = 5
        return {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


def test_collate_fn_wraps_each_example_as_a_system_and_user_message(tiny_pairwise_examples):
    tokenizer = _FakeTokenizer()
    collate_fn = make_hf_collate_fn(tokenizer, max_seq_len=16)
    batch = tiny_pairwise_examples[:2]

    collate_fn(batch)

    conversations = tokenizer.last_call["conversations"]
    assert len(conversations) == 2
    for conversation, example in zip(conversations, batch, strict=True):
        assert conversation == [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": format_input(example)},
        ]


def test_collate_fn_requests_tokenized_padded_truncated_tensors(tiny_pairwise_examples):
    tokenizer = _FakeTokenizer()
    collate_fn = make_hf_collate_fn(tokenizer, max_seq_len=16)

    collate_fn(tiny_pairwise_examples[:2])

    call = tokenizer.last_call
    assert call["tokenize"] is True
    assert call["add_generation_prompt"] is True
    assert call["padding"] is True
    assert call["truncation"] is True
    assert call["max_length"] == 16
    assert call["return_tensors"] == "pt"
    assert call["return_dict"] is True


def test_collate_fn_output_has_required_keys_and_shapes(tiny_pairwise_examples):
    tokenizer = _FakeTokenizer()
    collate_fn = make_hf_collate_fn(tokenizer, max_seq_len=16)
    batch = tiny_pairwise_examples[:3]

    result = collate_fn(batch)

    assert result["input_ids"].shape == (3, 5)
    assert result["attention_mask"].shape == (3, 5)
    assert result["labels"].tolist() == [e.label for e in batch]
    assert result["labels"].dtype == torch.long


def test_collate_fn_raises_on_empty_batch():
    collate_fn = make_hf_collate_fn(_FakeTokenizer(), max_seq_len=16)
    with pytest.raises(CheckFailed, match="empty batch"):
        collate_fn([])
