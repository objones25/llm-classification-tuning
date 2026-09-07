from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, Qwen2Config

from llm_reward.models import sft_head
from llm_reward.models.config import SFTHeadConfig
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


def _config(tmp_path, **overrides: Any) -> SFTHeadConfig:
    # dict[str, Any]: splatted into SFTHeadConfig's constructor, which has a different field type
    # per key -- a concrete value type here would make Pyright check every field against one
    # uniform type instead.
    kwargs: dict[str, Any] = dict(
        seed=1, batch_size=2, epochs=1, lr=1e-4, output_dir=tmp_path, run_name="r",
        hf_model_name="unused-because-mocked", max_seq_len=16,
    )
    kwargs.update(overrides)
    return SFTHeadConfig(**kwargs)


def test_build_model_returns_working_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda _name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path))

    assert isinstance(bundle, ModelBundle)
    input_ids = torch.randint(1, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_one_training_step_lowers_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda _name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path, lr=1e-2))
    input_ids = torch.randint(1, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    labels = torch.tensor([0, 1])

    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=1e-2)
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


def test_head_lr_creates_two_param_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda _name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path, head_lr=1e-2))
    assert bundle.param_groups is not None
    assert len(bundle.param_groups) == 2
    assert bundle.param_groups[1]["lr"] == 1e-2


def test_head_lr_none_leaves_param_groups_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda _name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path))
    assert bundle.param_groups is None


def test_real_model_is_reachable_through_the_wrapper_for_push_to_hub_dispatch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda _name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path))
    from transformers import PreTrainedModel

    assert isinstance(bundle.model.hf_model, PreTrainedModel)


def test_gradient_checkpointing_enables_input_require_grads_when_backbone_frozen(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda _name: _FakeTokenizer())

    bundle = sft_head.build_model(
        _config(tmp_path, gradient_checkpointing=True, freeze_backbone=True)
    )
    # ModelBundle.model is typed as the general nn.Module (the registry's DRY seam), so `.hf_model`
    # (a LogitsOnly-only attribute) isn't statically visible and Pyright falls back to
    # nn.Module.__getattr__'s stubbed `Tensor | Module` return type -- a stub gap, not a real bug.
    assert bundle.model.hf_model.is_gradient_checkpointing  # pyright: ignore[reportAttributeAccessIssue]
