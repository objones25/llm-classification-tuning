from pathlib import Path
from typing import Any

import torch

from llm_reward.models import lstm_baseline
from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle, build_model


def _config(**overrides: Any) -> LSTMConfig:
    # dict[str, Any]: splatted into LSTMConfig's constructor, which has a different field type
    # per key -- a concrete value type here would make Pyright check every field against one
    # uniform type instead.
    kwargs: dict[str, Any] = dict(
        seed=1, batch_size=2, epochs=1, lr=1e-2, output_dir=Path("out"), run_name="r",
        vocab_size=200, max_seq_len=8,
    )
    kwargs.update(overrides)
    return LSTMConfig(**kwargs)


def test_build_model_returns_a_model_bundle():
    bundle = lstm_baseline.build_model(_config())
    assert isinstance(bundle, ModelBundle)
    assert bundle.param_groups is None


def test_registry_dispatches_to_lstm_baseline():
    bundle = build_model(_config())
    assert isinstance(bundle, ModelBundle)


def test_collate_fn_output_has_required_keys_and_shapes(tiny_pairwise_examples):
    config = _config()
    bundle = lstm_baseline.build_model(config)
    batch = bundle.collate_fn(tiny_pairwise_examples[:4])
    assert "labels" in batch
    assert batch["token_ids"].shape == (4, config.max_seq_len)
    assert batch["labels"].shape == (4,)
    assert batch["labels"].dtype == torch.long


def test_forward_produces_correct_shape_and_finite_values(tiny_pairwise_examples):
    config = _config()
    bundle = lstm_baseline.build_model(config)
    batch = bundle.collate_fn(tiny_pairwise_examples[:4])
    logits = bundle.model(batch["token_ids"])
    assert logits.shape == (4, 3)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_one_training_step_lowers_loss(tiny_pairwise_examples):
    config = _config(lr=0.1)
    bundle = lstm_baseline.build_model(config)
    batch = bundle.collate_fn(tiny_pairwise_examples[:4])
    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=config.lr)

    # eval() for both loss measurements -- dropout is stochastic, so comparing a train()-mode
    # loss before against another train()-mode loss after would compare two different random
    # masks, not the effect of the optimizer step.
    bundle.model.eval()
    loss_before = torch.nn.functional.cross_entropy(
        bundle.model(batch["token_ids"]), batch["labels"]
    )
    bundle.model.train()
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    bundle.model.eval()
    with torch.no_grad():
        loss_after = torch.nn.functional.cross_entropy(
            bundle.model(batch["token_ids"]), batch["labels"]
        )
    assert loss_after.item() < loss_before.item()


def test_dropout_is_active_in_train_mode_and_off_in_eval_mode(tiny_pairwise_examples):
    config = _config(dropout=0.5)
    bundle = lstm_baseline.build_model(config)
    batch = bundle.collate_fn(tiny_pairwise_examples[:4])

    bundle.model.train()
    with torch.no_grad():
        train_a = bundle.model(batch["token_ids"])
        train_b = bundle.model(batch["token_ids"])
    assert not torch.equal(train_a, train_b)  # different random dropout masks each call

    bundle.model.eval()
    with torch.no_grad():
        eval_a = bundle.model(batch["token_ids"])
        eval_b = bundle.model(batch["token_ids"])
    torch.testing.assert_close(eval_a, eval_b)  # dropout off -- deterministic


def test_all_parameters_receive_gradients(tiny_pairwise_examples):
    config = _config()
    bundle = lstm_baseline.build_model(config)
    batch = bundle.collate_fn(tiny_pairwise_examples[:4])
    loss = torch.nn.functional.cross_entropy(bundle.model(batch["token_ids"]), batch["labels"])
    loss.backward()
    dead = [
        name for name, p in bundle.model.named_parameters()
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))
    ]
    assert dead == []
