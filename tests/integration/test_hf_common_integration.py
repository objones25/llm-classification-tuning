import pytest
from transformers import AutoTokenizer

from llm_reward.models._hf_common import SYSTEM_MESSAGE, make_hf_collate_fn


@pytest.mark.integration
def test_real_chat_template_produces_special_tokens_and_system_message(tiny_pairwise_examples):
    """Downloads the real Qwen/Qwen2.5-0.5B-Instruct tokenizer (small, fast) rather than the
    full 7B model this variant trains on -- this only needs to verify tokenizer/chat-template
    behavior, not exercise the whole model."""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = make_hf_collate_fn(tokenizer, max_seq_len=128)

    batch = collate_fn(tiny_pairwise_examples[:2])

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for row in batch["input_ids"]:
        assert im_start_id in row.tolist()
        assert im_end_id in row.tolist()

    decoded = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
    assert SYSTEM_MESSAGE in decoded
    # Not asserting this is the literal suffix of the decoded string: with padding=True over a
    # batch of differently-sized examples, the shorter row gets pad tokens appended after it.
    assert "<|im_start|>assistant" in decoded
