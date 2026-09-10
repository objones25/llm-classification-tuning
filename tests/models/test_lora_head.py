
from typing import Any

import pytest
import torch
from transformers import AutoModelForSequenceClassification, Qwen2Config

from llm_reward.models import lora_head
from llm_reward.models.config import LoRAConfig
from llm_reward.models.registry import ModelBundle, load_model_state_dict, model_state_dict


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


def _config(tmp_path, **overrides: Any) -> LoRAConfig:
    # dict[str, Any]: splatted into LoRAConfig's constructor, which has a different field type
    # per key -- a concrete value type here would make Pyright check every field against one
    # uniform type instead.
    kwargs: dict[str, Any] = dict(
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
    # ModelBundle.model is typed as the general nn.Module (the registry's DRY seam), so `.hf_model`
    # (a LogitsOnly-only attribute) isn't statically visible and Pyright falls back to
    # nn.Module.__getattr__'s stubbed `Tensor | Module` return type -- a stub gap, not a real bug.
    trainable = [
        n for n, p in bundle.model.hf_model.named_parameters()  # pyright: ignore[reportAttributeAccessIssue]
        if p.requires_grad
    ]
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
    # See the pyright: ignore comment on test_backbone_is_frozen_except_lora_and_head above --
    # same nn.Module.__getattr__ stub gap for the type-erased ModelBundle.model.hf_model access.
    assert bundle.model.hf_model.is_gradient_checkpointing  # pyright: ignore[reportAttributeAccessIssue]


def test_bundle_supplies_lora_state_dict_hooks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path))
    assert bundle.state_dict_fn is not None
    assert bundle.load_state_dict_fn is not None


@pytest.mark.filterwarnings("ignore:Could not find a config file")
def test_model_state_dict_is_much_smaller_than_the_full_model(tmp_path, monkeypatch):
    # get_peft_model_state_dict tries to detect whether the tokenizer's vocab was resized by
    # reading the base model's saved config file -- these tests build the base model via
    # from_config() (no directory on disk), which the real code path never does (it always
    # uses from_pretrained(hf_model_name)); harmless in both cases, but only the test path
    # trips the warning pytest's filterwarnings=["error"] turns into a failure.
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path))
    saved = model_state_dict(bundle)
    full = bundle.model.hf_model.state_dict()  # pyright: ignore[reportAttributeAccessIssue]

    saved_params = sum(v.numel() for v in saved.values())
    full_params = sum(v.numel() for v in full.values())
    # The whole point of the fix: this must not be "basically everything".
    assert saved_params < full_params / 2


@pytest.mark.filterwarnings("ignore:Could not find a config file")
def test_save_then_load_reconstructs_identical_outputs_on_the_same_base(tmp_path, monkeypatch):
    """The real-world case: build_model always reloads the SAME hf_model_name, so both bundles
    share an identical frozen base -- only the trainable adapter+head weights should need to
    round-trip through the checkpoint for outputs to match exactly."""
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    torch.manual_seed(42)
    bundle1 = lora_head.build_model(_config(tmp_path))
    optimizer = torch.optim.AdamW(
        [p for p in bundle1.model.parameters() if p.requires_grad], lr=0.1
    )
    input_ids = torch.randint(1, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    for _ in range(5):
        loss = bundle1.model(input_ids=input_ids, attention_mask=attention_mask).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    bundle1.model.eval()
    saved = model_state_dict(bundle1)

    torch.manual_seed(42)  # same seed -> identical frozen base as bundle1's initial state
    bundle2 = lora_head.build_model(_config(tmp_path))
    bundle2.model.eval()

    with torch.no_grad():
        expected = bundle1.model(input_ids=input_ids, attention_mask=attention_mask)
        before_load = bundle2.model(input_ids=input_ids, attention_mask=attention_mask)
    load_model_state_dict(bundle2, saved)
    with torch.no_grad():
        after_load = bundle2.model(input_ids=input_ids, attention_mask=attention_mask)

    assert not torch.allclose(before_load, expected, atol=1e-4)
    assert torch.allclose(after_load, expected, atol=1e-4)


def test_load_model_state_dict_accepts_a_pre_fix_full_checkpoint(tmp_path, monkeypatch):
    """Backward compatibility: a checkpoint saved before this fix holds bundle.model's full
    state_dict(), not the adapter-only dict this fix now produces. Loading one must not raise."""
    monkeypatch.setattr(
        lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model
    )
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    torch.manual_seed(7)
    bundle1 = lora_head.build_model(_config(tmp_path))
    bundle1.model.eval()
    old_format_full_state_dict = bundle1.model.state_dict()

    torch.manual_seed(7)
    bundle2 = lora_head.build_model(_config(tmp_path))
    bundle2.model.eval()

    load_model_state_dict(bundle2, old_format_full_state_dict)  # must not raise

    input_ids = torch.randint(1, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    with torch.no_grad():
        out1 = bundle1.model(input_ids=input_ids, attention_mask=attention_mask)
        out2 = bundle2.model(input_ids=input_ids, attention_mask=attention_mask)
    assert torch.allclose(out1, out2, atol=1e-5)
