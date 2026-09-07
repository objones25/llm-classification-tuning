
import torch
from transformers import AutoModelForSequenceClassification, Qwen2Config

from llm_reward.models import lora_head
from llm_reward.models.config import LoRAConfig
from llm_reward.models.registry import ModelBundle


def _tiny_qwen2_model(*args, **kwargs):
    cfg = Qwen2Config(
        vocab_size=1000, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0,
    )
    cfg.num_labels = 3
    return AutoModelForSequenceClassification.from_config(cfg)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"


def _config(tmp_path, **overrides) -> LoRAConfig:
    kwargs = dict(
        seed=1, batch_size=2, epochs=1, lr=1e-3, output_dir=tmp_path, run_name="r",
        hf_model_name="unused-because-mocked", max_seq_len=16,
        lora_rank=4, lora_alpha=8,
    )
    kwargs.update(overrides)
    return LoRAConfig(**kwargs)


def test_build_model_returns_working_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path))

    assert isinstance(bundle, ModelBundle)
    input_ids = torch.randint(1, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_backbone_is_frozen_except_lora_and_head(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path))
    trainable = [n for n, p in bundle.model.hf_model.named_parameters() if p.requires_grad]
    assert trainable  # something is trainable
    assert all("lora_" in n or "modules_to_save" in n for n in trainable)


def test_one_training_step_lowers_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path, lr=1e-2))
    input_ids = torch.randint(1, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    labels = torch.tensor([0, 1])

    trainable_params = [p for p in bundle.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-2)
    loss_before = torch.nn.functional.cross_entropy(
        bundle.model(input_ids=input_ids, attention_mask=attention_mask), labels
    )
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    with torch.no_grad():
        loss_after = torch.nn.functional.cross_entropy(
            bundle.model(input_ids=input_ids, attention_mask=attention_mask), labels
        )
    assert loss_after.item() < loss_before.item()


def test_head_lr_splits_adapter_and_head_into_two_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path, head_lr=1e-2))
    assert bundle.param_groups is not None
    assert len(bundle.param_groups) == 2
    assert bundle.param_groups[1]["lr"] == 1e-2


def test_real_model_is_reachable_through_the_wrapper_for_push_to_hub_dispatch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path))
    from peft import PeftModel

    assert isinstance(bundle.model.hf_model, PeftModel)


def test_gradient_checkpointing_enables_input_require_grads(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path, gradient_checkpointing=True))
    assert bundle.model.hf_model.is_gradient_checkpointing
