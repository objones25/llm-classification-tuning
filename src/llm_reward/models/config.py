from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import ClassVar, Literal

import yaml

from ..negative_space import require


@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    variant: ClassVar[str]
    seed: int
    batch_size: int
    epochs: int
    lr: float
    output_dir: Path
    run_name: str
    val_fraction: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    lr_scheduler: Literal["constant", "linear", "cosine"] = "linear"
    class_weights: tuple[float, float, float] | None = None
    mixed_precision: Literal["no", "bf16"] = "no"


@dataclass(frozen=True, kw_only=True)
class LSTMConfig(TrainConfig):
    variant: ClassVar[str] = "lstm_baseline"
    max_seq_len: int = 256
    vocab_size: int = 30_000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2


@dataclass(frozen=True, kw_only=True)
class SFTHeadConfig(TrainConfig):
    variant: ClassVar[str] = "small_sft_head"
    hf_model_name: str
    max_seq_len: int
    freeze_backbone: bool = False
    head_lr: float | None = None
    gradient_checkpointing: bool = False


@dataclass(frozen=True, kw_only=True)
class LoRAConfig(TrainConfig):
    variant: ClassVar[str] = "medium_lora"
    hf_model_name: str
    max_seq_len: int
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    head_lr: float | None = None
    gradient_checkpointing: bool = False


_VARIANTS: dict[str, type[TrainConfig]] = {
    LSTMConfig.variant: LSTMConfig,
    SFTHeadConfig.variant: SFTHeadConfig,
    LoRAConfig.variant: LoRAConfig,
}


class ConfigError(Exception):
    """Raised for a malformed config YAML: unknown variant, or a field that doesn't belong to
    the chosen subclass. Operating error — a bad file on disk — not a programmer error."""


def load_config(yaml_path: Path) -> TrainConfig:
    require(yaml_path.exists(), f"config file not found: {yaml_path}")
    with yaml_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    require(isinstance(raw, dict), f"{yaml_path} must contain a YAML mapping")

    variant = raw.pop("variant", None)
    if variant not in _VARIANTS:
        raise ConfigError(f"{yaml_path}: variant {variant!r} is not one of {sorted(_VARIANTS)}")
    config_cls = _VARIANTS[variant]

    if "output_dir" in raw:
        raw["output_dir"] = Path(raw["output_dir"])
    if "target_modules" in raw:
        raw["target_modules"] = tuple(raw["target_modules"])
    if raw.get("class_weights") is not None:
        raw["class_weights"] = tuple(raw["class_weights"])

    valid_fields = {f.name for f in fields(config_cls)}
    unknown = set(raw) - valid_fields
    if unknown:
        raise ConfigError(f"{yaml_path}: unknown field(s) for {variant!r}: {sorted(unknown)}")

    return config_cls(**raw)
