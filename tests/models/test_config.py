from pathlib import Path

import pytest

from llm_reward.models.config import (
    ConfigError,
    LoRAConfig,
    LSTMConfig,
    SFTHeadConfig,
    TrainConfig,
    load_config,
)


def _base_kwargs(**overrides):
    kwargs = dict(seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r")
    kwargs.update(overrides)
    return kwargs


def test_lstm_config_constructs_with_defaults():
    config = LSTMConfig(**_base_kwargs())
    assert config.variant == "lstm_baseline"
    assert config.max_seq_len == 256
    assert isinstance(config, TrainConfig)


def test_sft_head_config_requires_hf_model_name_and_max_seq_len():
    config = SFTHeadConfig(**_base_kwargs(), hf_model_name="Qwen/Qwen2.5-0.5B-Instruct", max_seq_len=1024)
    assert config.variant == "small_sft_head"
    assert config.head_lr is None


def test_lora_config_constructs_without_typeerror():
    # Regression test: as originally drafted (no kw_only), this line raised TypeError at
    # class-definition time because hf_model_name (no default) followed TrainConfig's defaulted
    # fields once dataclass inheritance flattened them.
    config = LoRAConfig(
        **_base_kwargs(), hf_model_name="Qwen/Qwen2.5-7B-Instruct", max_seq_len=2048,
    )
    assert config.lora_rank == 8
    assert config.target_modules == ("q_proj", "v_proj")


def test_mixed_precision_defaults_to_no():
    config = LSTMConfig(**_base_kwargs())
    assert config.mixed_precision == "no"


def test_gradient_checkpointing_defaults_to_false_on_hf_backed_variants():
    sft = SFTHeadConfig(**_base_kwargs(), hf_model_name="x", max_seq_len=8)
    lora = LoRAConfig(**_base_kwargs(), hf_model_name="x", max_seq_len=8)
    assert sft.gradient_checkpointing is False
    assert lora.gradient_checkpointing is False


def test_configs_are_frozen():
    config = LSTMConfig(**_base_kwargs())
    with pytest.raises(AttributeError):
        config.lr = 0.1


def test_two_configs_with_same_fields_are_equal():
    a = LSTMConfig(**_base_kwargs())
    b = LSTMConfig(**_base_kwargs())
    assert a == b


def test_configs_of_different_variants_are_never_equal():
    lstm = LSTMConfig(**_base_kwargs())
    sft = SFTHeadConfig(**_base_kwargs(), hf_model_name="x", max_seq_len=8)
    assert lstm != sft


def test_load_config_reads_lstm_yaml(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "variant: lstm_baseline\n"
        "seed: 1\nbatch_size: 8\nepochs: 2\nlr: 0.001\n"
        "output_dir: out\nrun_name: r\n"
    )
    config = load_config(yaml_path)
    assert isinstance(config, LSTMConfig)
    assert config.output_dir == Path("out")


def test_load_config_rejects_unknown_variant(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("variant: not_a_real_variant\nseed: 1\n")
    with pytest.raises(ConfigError, match="not_a_real_variant"):
        load_config(yaml_path)


def test_load_config_rejects_field_not_on_the_chosen_subclass(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "variant: lstm_baseline\n"
        "seed: 1\nbatch_size: 8\nepochs: 2\nlr: 0.001\n"
        "output_dir: out\nrun_name: r\n"
        "lora_rank: 8\n"  # belongs to LoRAConfig, not LSTMConfig
    )
    with pytest.raises(ConfigError, match="lora_rank"):
        load_config(yaml_path)
