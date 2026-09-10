from pathlib import Path

import pytest
import torch
from torch import nn

from llm_reward.models.registry import (
    ModelBundle,
    build_model,
    load_model_state_dict,
    model_state_dict,
    register,
)
from llm_reward.negative_space import CheckFailed


def test_register_then_build_model_dispatches_correctly():
    from dataclasses import dataclass
    from typing import ClassVar

    from llm_reward.models.config import TrainConfig

    @dataclass(frozen=True, kw_only=True)
    class _TestConfig(TrainConfig):
        variant: ClassVar[str] = "test_variant_a"

    @register("test_variant_a")
    def _build(config):
        return ModelBundle(model=nn.Linear(1, 3), collate_fn=lambda batch: {})

    config = _TestConfig(
        seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r"
    )
    bundle = build_model(config)
    assert isinstance(bundle, ModelBundle)
    assert isinstance(bundle.model, nn.Linear)


def test_build_model_raises_on_unknown_variant():
    from dataclasses import dataclass
    from typing import ClassVar

    from llm_reward.models.config import TrainConfig

    @dataclass(frozen=True, kw_only=True)
    class _UnregisteredConfig(TrainConfig):
        variant: ClassVar[str] = "never_registered"

    config = _UnregisteredConfig(
        seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r"
    )
    with pytest.raises(CheckFailed, match="unknown variant"):
        build_model(config)


def test_register_rejects_duplicate_variant_name():
    @register("test_variant_b")
    def _build_one(config):
        return ModelBundle(model=nn.Linear(1, 3), collate_fn=lambda batch: {})

    with pytest.raises(CheckFailed, match="already registered"):
        @register("test_variant_b")
        def _build_two(config):
            return ModelBundle(model=nn.Linear(1, 3), collate_fn=lambda batch: {})


def test_model_bundle_param_groups_defaults_to_none():
    bundle = ModelBundle(model=nn.Linear(1, 3), collate_fn=lambda batch: {})
    assert bundle.param_groups is None


def test_model_bundle_state_dict_hooks_default_to_none():
    bundle = ModelBundle(model=nn.Linear(1, 3), collate_fn=lambda batch: {})
    assert bundle.state_dict_fn is None
    assert bundle.load_state_dict_fn is None


def test_model_state_dict_falls_back_to_plain_state_dict_when_no_hook():
    model = nn.Linear(2, 3)
    bundle = ModelBundle(model=model, collate_fn=lambda batch: {})
    state_dict = model_state_dict(bundle)
    assert set(state_dict) == {"weight", "bias"}
    assert torch.equal(state_dict["weight"], model.weight)


def test_load_model_state_dict_falls_back_to_plain_load_when_no_hook():
    model = nn.Linear(2, 3)
    bundle = ModelBundle(model=model, collate_fn=lambda batch: {})
    new_weight = torch.randn_like(model.weight)
    load_model_state_dict(bundle, {"weight": new_weight, "bias": model.bias.clone()})
    assert torch.equal(model.weight, new_weight)


def test_model_state_dict_uses_custom_hook_when_provided():
    model = nn.Linear(2, 3)
    calls = []

    def fake_state_dict_fn(m):
        calls.append(m)
        return {"custom": torch.zeros(1)}

    bundle = ModelBundle(model=model, collate_fn=lambda batch: {}, state_dict_fn=fake_state_dict_fn)
    result = model_state_dict(bundle)
    assert calls == [model]
    assert result == {"custom": torch.zeros(1)}


def test_load_model_state_dict_uses_custom_hook_when_provided():
    model = nn.Linear(2, 3)
    calls = []

    def fake_load_fn(m, state_dict):
        calls.append((m, state_dict))

    bundle = ModelBundle(
        model=model, collate_fn=lambda batch: {}, load_state_dict_fn=fake_load_fn
    )
    sentinel = {"custom": torch.zeros(1)}
    load_model_state_dict(bundle, sentinel)
    assert calls == [(model, sentinel)]
