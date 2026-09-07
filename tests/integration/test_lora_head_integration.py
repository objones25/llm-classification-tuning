import pytest
import torch

from llm_reward.models.config import LoRAConfig
from llm_reward.models.lora_head import build_model


@pytest.mark.integration
def test_real_tokenizer_produces_expected_batch_keys(tmp_path, tiny_pairwise_examples):
    """Downloads the real Qwen/Qwen2.5-7B-Instruct tokenizer on first run (cached after)."""
    config = LoRAConfig(
        seed=1, batch_size=2, epochs=1, lr=1e-4, output_dir=tmp_path, run_name="r",
        hf_model_name="Qwen/Qwen2.5-7B-Instruct", max_seq_len=64,
    )
    bundle = build_model(config)
    batch = bundle.collate_fn(tiny_pairwise_examples[:2])
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert isinstance(batch["labels"], torch.Tensor)
