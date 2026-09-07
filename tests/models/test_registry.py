from pathlib import Path

import pytest
from torch import nn

from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle, build_model, register
from llm_reward.negative_space import CheckFailed


def _config():
    return LSTMConfig(seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r")


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

    config = _TestConfig(seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r")
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

    config = _UnregisteredConfig(seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r")
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
