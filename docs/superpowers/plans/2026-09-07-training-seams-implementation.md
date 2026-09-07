# Training Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the data pipeline, model registry, train/eval loop, and Hugging Face Hub
persistence for the LLM Classification Finetuning competition, exactly as specified.

**Architecture:** A `PairwiseExample` dataclass and a `ModelBundle`/`registry.build_model`
dispatch are the two seams that let `data/`, three independent model variants
(`lstm_baseline.py`/`sft_head.py`/`lora_head.py`), and `train.py`/`evaluate.py` be built in
parallel by different sessions, touching disjoint files. `scripts/push_to_hub.py` persists a
reviewed run to the Hub since the RunPod pod's local disk does not survive pod deletion.

**Tech Stack:** Python 3.13, uv, PyTorch 2.8.0, transformers, peft, huggingface_hub, PyYAML,
wandb, pytest (strict mode, already configured in `pyproject.toml`).

**Spec:** `docs/superpowers/specs/2026-09-07-training-seams-design.md` (read Sections A-E plus all
2026-09-07 amendments — the `kw_only=True` dataclass fix, the `_LogitsOnly` wrapper, and the
device-placement/mixed-precision/W&B-metrics amendment added after a pytorch-skill review of this
plan, before any task in this plan existed in its current form — before starting any task; this
plan assumes all of them).

## Global Constraints

- All `TrainConfig` dataclasses use `@dataclass(frozen=True, kw_only=True)` — without `kw_only`,
  `SFTHeadConfig`/`LoRAConfig` raise `TypeError` at class-definition time (verified in the spec).
- `ModelBundle.model.forward(**inputs)` must return a bare `[B, 3]` float tensor, never a
  `ModelOutput`/dict. HF-backed variants wrap their real model in `_LogitsOnly` to guarantee this.
- Every negative-space check uses `require`/`bounded`/`check_shape`/`check_finite` from
  `src/llm_reward/negative_space.py` (Task 1), never a bare `assert` — `python -O` strips `assert`
  silently, `require` does not.
- Programmer errors (`require(...)` fails) crash. Operating errors (bad YAML, missing file,
  mismatched resume config) raise a typed exception and are handled at the call site — never both.
- pytest config already in `pyproject.toml`: `-m "not slow and not integration and not gpu"` by
  default, `--import-mode=importlib`, `filterwarnings = ["error"]`. Every test file in this plan
  that needs real network (HF tokenizer/model download, kagglehub, wandb) is `@pytest.mark.integration`
  and lives under `tests/integration/` so the root conftest's directory-based auto-marking catches
  it even if the decorator is forgotten.
- Model choices (already decided, not open for reinterpretation): small variant =
  `Qwen/Qwen2.5-0.5B-Instruct`, medium variant = `Qwen/Qwen2.5-7B-Instruct`. Both are ungated on
  the Hub — no HF license click-through needed before `HF_TOKEN` can pull them.
- `train.py` (Task 15) resolves `device` once via `torch.accelerator.current_accelerator(check_available=True)
  or torch.device("cpu")`, moves `bundle.model.to(device)` once, and moves every batch tensor to
  `device` before the forward pass. This is not optional or deferrable — without it, training
  silently runs on CPU regardless of what hardware it's launched on (verified against the pytorch
  skill's `05-data-training-loop.md`).
- `wandb.init(...)`'s return value (the `run` object) is held and used for every later call —
  `run.log(...)`, `run.id`, `run.define_metric(...)` — never the bare `wandb.*` module-level form
  (verified against wandb's own docs and the observability-expert skill's `WB004` rule). A test
  double standing in for the `wandb` module must return an object supporting this same shape.
- Every task's tests run with `uv run pytest <path> -v` and must pass before the task's commit.

---

## Track 0 — Shared Contracts (sequential; every other track depends on this one)

### Task 1: Negative-space helper module

**Files:**
- Create: `src/llm_reward/negative_space.py`
- Test: `tests/test_negative_space.py`

**Interfaces:**
- Produces: `require(condition, message="")`, `unreachable(message="")`,
  `bounded(iterable, limit, name="loop")`, `check_shape(array, spec, name="array")`,
  `check_finite(value, name="value")`, `CheckFailed(AssertionError)` — every later task imports
  from here instead of using bare `assert`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_negative_space.py
import pytest

from llm_reward.negative_space import CheckFailed, bounded, check_finite, check_shape, require


def test_require_passes_silently_when_true():
    require(1 < 2)


def test_require_raises_check_failed_with_message():
    with pytest.raises(CheckFailed, match="ordering broken"):
        require(2 < 1, "ordering broken")


def test_bounded_yields_items_under_the_limit():
    assert list(bounded(range(3), limit=5)) == [0, 1, 2]


def test_bounded_raises_once_the_limit_is_exceeded():
    with pytest.raises(CheckFailed, match="exceeded its bound of 3"):
        list(bounded(range(10), limit=3, name="retries"))


def test_check_shape_returns_named_dim_bindings():
    class _Fake:
        shape = (2, 3)

    assert check_shape(_Fake(), (2, "n"), name="x") == {"n": 3}


def test_check_finite_raises_on_nan():
    with pytest.raises(CheckFailed, match="not finite"):
        check_finite(float("nan"), name="loss")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_negative_space.py -v`
Expected: FAIL (collection error) — `ModuleNotFoundError: No module named 'llm_reward.negative_space'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/negative_space.py
"""Runtime checks that survive `python -O`. Standard library only.

Use for programmer errors (a state your own code should have made impossible).
For operating errors — bad user input, missing file, network failure — raise a
typed exception instead and handle it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, NoReturn, TypeVar

__all__ = ["CheckFailed", "require", "unreachable", "bounded", "check_shape", "check_finite"]

T = TypeVar("T")


class CheckFailed(AssertionError):
    """A contract was violated. Subclasses AssertionError so pytest.raises(AssertionError) works."""


def require(condition: object, message: str = "") -> None:
    if not condition:
        raise CheckFailed(message or "requirement failed")


def unreachable(message: str = "") -> NoReturn:
    raise CheckFailed(message or "reached unreachable code")


def bounded(iterable: Iterable[T], limit: int, name: str = "loop") -> Iterator[T]:
    require(limit >= 1, f"{name}: bound must be at least 1, got {limit}")
    count = 0
    for item in iterable:
        count += 1
        if count > limit:
            raise CheckFailed(f"{name} exceeded its bound of {limit} iterations")
        yield item


def check_shape(array: Any, spec: Sequence[object], name: str = "array") -> dict[str, int]:
    shape = getattr(array, "shape", None)
    require(shape is not None, f"{name} has no .shape attribute")
    shape = tuple(int(d) for d in shape)
    require(
        len(shape) == len(spec),
        f"{name}: expected {len(spec)} dims {tuple(spec)}, got {len(shape)} {shape}",
    )
    bindings: dict[str, int] = {}
    for axis, (actual, expected) in enumerate(zip(shape, spec)):
        if expected is None or expected == -1:
            continue
        if isinstance(expected, str):
            if expected in bindings:
                require(
                    bindings[expected] == actual,
                    f"{name}: dim '{expected}' is {bindings[expected]} elsewhere but {actual} "
                    f"at axis {axis}; full shape {shape}",
                )
            else:
                bindings[expected] = actual
        else:
            require(
                actual == int(expected),
                f"{name}: axis {axis} expected {expected}, got {actual}; full shape {shape}",
            )
    return bindings


def check_finite(value: Any, name: str = "value") -> None:
    isfinite = getattr(value, "isfinite", None)
    if callable(isfinite):
        result = isfinite()
        allf = getattr(result, "all", None)
        ok = bool(allf()) if callable(allf) else bool(result)
        require(ok, f"{name} is not finite: {value}")
        return
    require(math.isfinite(float(value)), f"{name} is not finite: {value}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_negative_space.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/negative_space.py tests/test_negative_space.py
git commit -m "feat: add negative-space fail-fast helper module"
```

---

### Task 2: `PairwiseExample` dataclass

**Files:**
- Create: `src/llm_reward/data/__init__.py` (empty)
- Create: `src/llm_reward/data/pairwise.py`
- Test: `tests/data/test_pairwise.py`

**Interfaces:**
- Consumes: `require` from Task 1.
- Produces: `PairwiseExample(id: str, prompt: str, response_a: str, response_b: str, label: int)`
  — every later task imports this dataclass.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_pairwise.py
import pytest

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.negative_space import CheckFailed


def test_construct_valid_example():
    example = PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=0)
    assert example.label == 0
    assert example.id == "1"


@pytest.mark.parametrize("bad_label", [-1, 3, 99])
def test_rejects_out_of_range_label(bad_label):
    with pytest.raises(CheckFailed, match="label must be"):
        PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=bad_label)


def test_rejects_empty_id():
    with pytest.raises(CheckFailed, match="id must not be empty"):
        PairwiseExample(id="", prompt="p", response_a="a", response_b="b", label=0)


def test_is_frozen():
    example = PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=0)
    with pytest.raises(AttributeError):
        example.label = 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_pairwise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.data'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/data/__init__.py
```

```python
# src/llm_reward/data/pairwise.py
from __future__ import annotations

from dataclasses import dataclass

from ..negative_space import require


@dataclass(frozen=True)
class PairwiseExample:
    id: str
    prompt: str
    response_a: str
    response_b: str
    label: int  # 0=a, 1=b, 2=tie

    def __post_init__(self) -> None:
        require(self.label in (0, 1, 2), f"label must be 0, 1, or 2; got {self.label!r}")
        require(self.id, "id must not be empty")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_pairwise.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/data/__init__.py src/llm_reward/data/pairwise.py tests/data/test_pairwise.py
git commit -m "feat: add PairwiseExample, the shared data contract"
```

---

### Task 3: Root test infrastructure

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: `PairwiseExample` from Task 2.
- Produces: autouse RNG-seeding/network-blocking/cwd-isolation fixtures (apply to every test in
  the repo with no import needed), the `tiny_pairwise_examples` fixture (six hand-built records,
  two per label, reused by every later track instead of each hand-rolling its own), and the
  `integration`/`slow`/`gpu` marker registrations.

- [ ] **Step 1: Write the failing test** (this task's own test is that the fixture exists and the
  network guard actually blocks — write it first, confirm it fails for the right reason)

```python
# tests/test_conftest_fixtures.py
import socket

import pytest

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.negative_space import CheckFailed


def test_tiny_pairwise_examples_covers_all_three_labels(tiny_pairwise_examples):
    labels = {example.label for example in tiny_pairwise_examples}
    assert labels == {0, 1, 2}
    assert all(isinstance(example, PairwiseExample) for example in tiny_pairwise_examples)


def test_network_is_blocked_by_default():
    with pytest.raises(CheckFailed if False else Exception, match="network connection"):
        socket.create_connection(("example.com", 80))
```

(The `CheckFailed if False else Exception` is deliberate: the network guard's own exception type,
`NetworkBlockedError`, is defined inside `conftest.py` and this test file should not need to import
it — matching on the message is enough and keeps this test decoupled from where the guard lives.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conftest_fixtures.py -v`
Expected: FAIL — `fixture 'tiny_pairwise_examples' not found`

- [ ] **Step 3: Write the implementation**

```python
# tests/conftest.py
from __future__ import annotations

import random
import socket
import sys
from pathlib import Path

import pytest

from llm_reward.data.pairwise import PairwiseExample

SEED = 20260907


class NetworkBlockedError(RuntimeError):
    """Raised when a unit test tries to open a socket."""


@pytest.fixture(autouse=True)
def _deterministic_rngs() -> None:
    """Reseed every RNG before each test — per-test, not per-session, so a single test run in
    isolation gives the same numbers as a full suite run."""
    random.seed(SEED)
    import numpy as np

    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration"):
        return

    def guard(*args: object, **kwargs: object):
        raise NetworkBlockedError(
            "This test attempted a network connection. Stub the boundary with "
            "monkeypatch, or mark the test @pytest.mark.integration."
        )

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)


@pytest.fixture(autouse=True)
def _isolated_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", str(SEED))


@pytest.fixture
def tiny_pairwise_examples() -> list[PairwiseExample]:
    """Six hand-built records, two per label — the one shared fixture every track's tests build
    from, per CLAUDE.md's testing convention. Never hand-roll a second copy of this."""
    return [
        PairwiseExample(id="1", prompt="What is 2+2?", response_a="4", response_b="Four", label=0),
        PairwiseExample(
            id="2", prompt="Capital of France?",
            response_a="Paris is the capital.", response_b="I don't know.", label=0,
        ),
        PairwiseExample(id="3", prompt="Say hi", response_a="hi", response_b="Hello there!", label=1),
        PairwiseExample(
            id="4", prompt="Explain gravity",
            response_a="Things fall down.", response_b="Mass attracts mass.", label=1,
        ),
        PairwiseExample(id="5", prompt="Pick a number", response_a="7", response_b="7", label=2),
        PairwiseExample(
            id="6", prompt="Tell a joke",
            response_a="Why did the chicken cross the road?",
            response_b="Why did the chicken cross the road?", label=2,
        ),
    ]


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "slow: takes more than ~1s; deselected by default",
        "integration: crosses a process, service, or network boundary",
        "gpu: requires CUDA; skipped when unavailable",
    ):
        config.addinivalue_line("markers", line)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        parts = {p.name for p in Path(str(item.fspath)).parents}
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)


def pytest_report_header(config: pytest.Config) -> list[str]:
    return [f"llm-reward tests: seed={SEED} python={sys.version.split()[0]}"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_conftest_fixtures.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_conftest_fixtures.py
git commit -m "feat: add root test infrastructure (determinism, network guard, shared fixture)"
```

---

### Task 4: `models/config.py`

**Files:**
- Create: `src/llm_reward/models/__init__.py` (empty — Tasks 9/10/11 each append one import line)
- Create: `src/llm_reward/models/config.py`
- Test: `tests/models/test_config.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TrainConfig`, `LSTMConfig`, `SFTHeadConfig`, `LoRAConfig` (all
  `@dataclass(frozen=True, kw_only=True)`), `ConfigError(Exception)`, `load_config(yaml_path: Path) -> TrainConfig`.

- [ ] **Step 0: Add the one new dependency this task needs**

Run: `uv add pyyaml`

(`pyyaml` was already present transitively via other dependencies, but `config.py` imports it
directly — pin it explicitly rather than relying on an unrelated package's own dependency choice.)

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.models'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/models/__init__.py
```

```python
# src/llm_reward/models/config.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_config.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/llm_reward/models/__init__.py src/llm_reward/models/config.py tests/models/test_config.py
git commit -m "feat: add TrainConfig hierarchy and YAML loader"
```

---

### Task 5: `models/registry.py`

**Files:**
- Create: `src/llm_reward/models/registry.py`
- Test: `tests/models/test_registry.py`

**Interfaces:**
- Consumes: `PairwiseExample` (Task 2), `TrainConfig` (Task 4).
- Produces: `CollateFn`, `ModelBundle(model, collate_fn, param_groups=None)`,
  `register(variant) -> decorator`, `build_model(config: TrainConfig) -> ModelBundle`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_registry.py
from pathlib import Path

import pytest
import torch
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.models.registry'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/models/registry.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import TrainConfig

CollateFn = Callable[[list[PairwiseExample]], dict[str, torch.Tensor]]


@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module
    collate_fn: CollateFn
    param_groups: list[dict] | None = None


BuildFn = Callable[[TrainConfig], ModelBundle]
_REGISTRY: dict[str, BuildFn] = {}


def register(variant: str) -> Callable[[BuildFn], BuildFn]:
    def _decorator(fn: BuildFn) -> BuildFn:
        require(variant not in _REGISTRY, f"variant {variant!r} already registered")
        _REGISTRY[variant] = fn
        return fn

    return _decorator


def build_model(config: TrainConfig) -> ModelBundle:
    require(
        config.variant in _REGISTRY,
        f"unknown variant {config.variant!r}; registered: {sorted(_REGISTRY)}",
    )
    return _REGISTRY[config.variant](config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_registry.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/models/registry.py tests/models/test_registry.py
git commit -m "feat: add ModelBundle and the self-registering model registry"
```

---

### Task 6: `models/checkpoint.py`

**Files:**
- Create: `src/llm_reward/models/checkpoint.py`
- Test: `tests/models/test_checkpoint.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 4).
- Produces: `Checkpoint(epoch, global_step, model_state, optimizer_state, scheduler_state, best_val_metric, config, wandb_run_id)`,
  `ConfigMismatchError(Exception)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_checkpoint.py
from pathlib import Path

import torch

from llm_reward.models.checkpoint import Checkpoint, ConfigMismatchError
from llm_reward.models.config import LSTMConfig


def _config():
    return LSTMConfig(seed=1, batch_size=8, epochs=2, lr=1e-3, output_dir=Path("out"), run_name="r")


def test_checkpoint_round_trips_through_torch_save(tmp_path):
    checkpoint = Checkpoint(
        epoch=0,
        global_step=42,
        model_state={"weight": torch.zeros(2)},
        optimizer_state={},
        scheduler_state=None,
        best_val_metric=0.5,
        config=_config(),
        wandb_run_id="run-123",
    )
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    loaded: Checkpoint = torch.load(path, weights_only=False, map_location="cpu")

    assert loaded.epoch == 0
    assert loaded.global_step == 42
    assert loaded.config == _config()
    assert loaded.wandb_run_id == "run-123"
    torch.testing.assert_close(loaded.model_state["weight"], torch.zeros(2))


def test_config_mismatch_error_is_a_plain_exception():
    assert issubclass(ConfigMismatchError, Exception)
    assert not issubclass(ConfigMismatchError, AssertionError)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_checkpoint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.models.checkpoint'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/models/checkpoint.py
from __future__ import annotations

from dataclasses import dataclass

from .config import TrainConfig


@dataclass
class Checkpoint:
    epoch: int  # last COMPLETED epoch, 0-indexed
    global_step: int  # total training steps so far, across all epochs — carried across resume so
                       # the W&B step x-axis doesn't reset to 0 on a new process
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict | None
    best_val_metric: float
    config: TrainConfig
    wandb_run_id: str


class ConfigMismatchError(Exception):
    """Raised when a --resume checkpoint's config does not match the config just loaded from
    YAML. Operating error (someone edited the YAML between runs), not a programmer error."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_checkpoint.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/models/checkpoint.py tests/models/test_checkpoint.py
git commit -m "feat: add Checkpoint dataclass and ConfigMismatchError"
```

**Track 0 is now complete. Tracks A-F below all depend only on Tasks 1-6 and can proceed in any
order, in parallel, by separate sessions — each touches a disjoint set of files except the
single shared line each of Tasks 9/10/11 appends to `models/__init__.py`.**

---

## Track A — Data Pipeline (depends only on Track 0)

### Task 7: `load_pairwise_examples` and `split_train_val`

**Files:**
- Modify: `src/llm_reward/data/pairwise.py` (append to the file from Task 2)
- Create: `tests/fixtures/tiny_train.csv`
- Test: `tests/data/test_pairwise_loading.py`

**Interfaces:**
- Consumes: `PairwiseExample`, `require`, `bounded`.
- Produces: `load_pairwise_examples(csv_path: Path) -> list[PairwiseExample]`,
  `split_train_val(examples, val_fraction, seed) -> tuple[list, list]`.

**Note on the real competition CSV format**: `train.csv`'s `prompt`/`response_a`/`response_b`
columns store one or more conversation turns as a Python-list-literal string (e.g.
`'["What is 2+2?"]'`), per the competition's own description ("a judge provides one or more
prompts"). `_extract_text` below handles both that case and a plain string, defensively, so it
works whichever shape a given file turns out to use — **the first time `scripts/download_data.py`
(Task 8) produces a real file, inspect a few real rows and confirm this still matches; adjust
`_extract_text` if not.**

- [ ] **Step 1: Write the failing test**

```python
# tests/fixtures/tiny_train.csv
id,model_a,model_b,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_model_tie
1,gpt-x,claude-y,"What is 2+2?","4","Four",1,0,0
2,gpt-x,claude-y,"['Say hi', 'again']","['hi', 'hi again']","['hello', 'hello again']",0,1,0
3,gpt-x,claude-y,"Pick a number","7","7",0,0,1
```

```python
# tests/data/test_pairwise_loading.py
from pathlib import Path

import pytest

from llm_reward.data.pairwise import load_pairwise_examples, split_train_val
from llm_reward.negative_space import CheckFailed

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tiny_train.csv"


def test_load_pairwise_examples_parses_all_rows():
    examples = load_pairwise_examples(FIXTURE)
    assert len(examples) == 3
    assert [e.label for e in examples] == [0, 1, 2]


def test_load_pairwise_examples_joins_list_encoded_turns():
    examples = load_pairwise_examples(FIXTURE)
    multi_turn = next(e for e in examples if e.id == "2")
    assert "Say hi" in multi_turn.prompt
    assert "again" in multi_turn.prompt


def test_load_pairwise_examples_raises_on_missing_file(tmp_path):
    with pytest.raises(CheckFailed, match="not found"):
        load_pairwise_examples(tmp_path / "does_not_exist.csv")


def test_split_train_val_is_disjoint_and_covers_everything():
    examples = load_pairwise_examples(FIXTURE) * 4  # 12 examples, still 3 unique-content rows
    examples = [
        e.__class__(id=str(i), prompt=e.prompt, response_a=e.response_a, response_b=e.response_b, label=e.label)
        for i, e in enumerate(examples)
    ]
    train, val = split_train_val(examples, val_fraction=0.25, seed=1)
    assert len(train) + len(val) == len(examples)
    assert {e.id for e in train}.isdisjoint({e.id for e in val})


def test_split_train_val_rejects_out_of_range_fraction():
    from llm_reward.data.pairwise import PairwiseExample

    examples = [PairwiseExample(id="1", prompt="p", response_a="a", response_b="b", label=0)] * 2
    with pytest.raises(CheckFailed, match="val_fraction"):
        split_train_val(examples, val_fraction=1.5, seed=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/data/test_pairwise_loading.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_pairwise_examples'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/llm_reward/data/pairwise.py
import ast
import csv
import random
from pathlib import Path

from ..negative_space import bounded


def _extract_text(value: str) -> str:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    if isinstance(parsed, list):
        return "\n".join(str(turn) for turn in parsed)
    return value


def load_pairwise_examples(csv_path: Path) -> list[PairwiseExample]:
    require(csv_path.exists(), f"csv file not found: {csv_path}")
    examples: list[PairwiseExample] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in bounded(reader, limit=1_000_000, name="csv rows"):
            if row["winner_model_a"] == "1":
                label = 0
            elif row["winner_model_b"] == "1":
                label = 1
            else:
                require(row["winner_model_tie"] == "1", f"row {row['id']} has no winner set")
                label = 2
            examples.append(
                PairwiseExample(
                    id=row["id"],
                    prompt=_extract_text(row["prompt"]),
                    response_a=_extract_text(row["response_a"]),
                    response_b=_extract_text(row["response_b"]),
                    label=label,
                )
            )
    require(len(examples) > 0, f"no rows parsed from {csv_path}")
    return examples


def split_train_val(
    examples: list[PairwiseExample], val_fraction: float, seed: int
) -> tuple[list[PairwiseExample], list[PairwiseExample]]:
    require(0.0 < val_fraction < 1.0, f"val_fraction must be strictly interior, got {val_fraction}")
    require(len(examples) >= 2, f"need at least 2 examples to split, got {len(examples)}")

    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val, train = shuffled[:n_val], shuffled[n_val:]

    require(len(train) + len(val) == len(examples), "split lost or duplicated examples")
    require(
        {e.id for e in train}.isdisjoint({e.id for e in val}),
        "train and val overlap",
    )
    return train, val
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/data/test_pairwise_loading.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/data/pairwise.py tests/data/test_pairwise_loading.py tests/fixtures/tiny_train.csv
git commit -m "feat: parse train.csv into PairwiseExample records with a fail-fast train/val split"
```

---

### Task 8: `data/dataset.py` and `data/download.py`

**Files:**
- Create: `src/llm_reward/data/dataset.py`
- Create: `src/llm_reward/data/download.py`
- Test: `tests/data/test_dataset.py`
- Test: `tests/data/test_download.py`
- Test (integration, needs `KAGGLE_API_TOKEN`): `tests/integration/test_download_real.py`

**Interfaces:**
- Consumes: `PairwiseExample`, `require`.
- Produces: `PairwiseDataset(examples)` (a `torch.utils.data.Dataset`),
  `download_competition_data(competition="llm-classification-finetuning") -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_dataset.py
import pytest
from torch.utils.data import DataLoader

from llm_reward.data.dataset import PairwiseDataset
from llm_reward.data.pairwise import PairwiseExample
from llm_reward.negative_space import CheckFailed


def test_len_matches_input_length(tiny_pairwise_examples):
    dataset = PairwiseDataset(tiny_pairwise_examples)
    assert len(dataset) == 6


def test_getitem_returns_the_raw_example_not_a_tensor(tiny_pairwise_examples):
    dataset = PairwiseDataset(tiny_pairwise_examples)
    assert dataset[0] is tiny_pairwise_examples[0]
    assert isinstance(dataset[0], PairwiseExample)


def test_rejects_empty_example_list():
    with pytest.raises(CheckFailed, match="at least one example"):
        PairwiseDataset([])


def test_works_with_a_real_dataloader_and_custom_collate_fn(tiny_pairwise_examples):
    dataset = PairwiseDataset(tiny_pairwise_examples)
    loader = DataLoader(dataset, batch_size=2, collate_fn=lambda batch: [e.id for e in batch])
    batches = list(loader)
    assert sum(len(b) for b in batches) == 6
```

```python
# tests/data/test_download.py
from pathlib import Path

from llm_reward.data import download


def test_download_competition_data_returns_a_path(monkeypatch):
    monkeypatch.setattr(download.kagglehub, "competition_download", lambda competition: "/fake/path")
    result = download.download_competition_data()
    assert result == Path("/fake/path")


def test_download_competition_data_passes_competition_slug_through(monkeypatch):
    captured = {}

    def _fake_download(competition):
        captured["competition"] = competition
        return "/fake/path"

    monkeypatch.setattr(download.kagglehub, "competition_download", _fake_download)
    download.download_competition_data(competition="a-different-slug")
    assert captured["competition"] == "a-different-slug"
```

```python
# tests/integration/test_download_real.py
import pytest

from llm_reward.data.download import download_competition_data


@pytest.mark.integration
def test_real_download_returns_an_existing_directory():
    """Requires KAGGLE_API_TOKEN in the environment (loaded from .env) and accepted competition
    rules on Kaggle. Run explicitly: uv run pytest -m integration tests/integration/test_download_real.py"""
    path = download_competition_data()
    assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_dataset.py tests/data/test_download.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/data/dataset.py
from __future__ import annotations

from torch.utils.data import Dataset

from ..negative_space import require
from .pairwise import PairwiseExample


class PairwiseDataset(Dataset):
    def __init__(self, examples: list[PairwiseExample]) -> None:
        require(len(examples) > 0, "PairwiseDataset requires at least one example")
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> PairwiseExample:
        return self._examples[idx]
```

```python
# src/llm_reward/data/download.py
from __future__ import annotations

from pathlib import Path

import kagglehub


def download_competition_data(competition: str = "llm-classification-finetuning") -> Path:
    return Path(kagglehub.competition_download(competition))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data/test_dataset.py tests/data/test_download.py -v`
Expected: PASS (6 tests). The integration test is skipped by default (`-m "not integration"`);
run it manually with `uv run pytest -m integration tests/integration/test_download_real.py -v`
once `KAGGLE_API_TOKEN` is available, and use its result to double-check Task 7's
`_extract_text` assumption against the real file.

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/data/dataset.py src/llm_reward/data/download.py \
        tests/data/test_dataset.py tests/data/test_download.py tests/integration/test_download_real.py
git commit -m "feat: add PairwiseDataset and the kagglehub download wrapper"
```

---

## Track B — LSTM Baseline (depends only on Track 0)

### Task 9: `models/lstm_baseline.py`

**Files:**
- Create: `src/llm_reward/models/lstm_baseline.py`
- Create: `configs/lstm_baseline.yaml`
- Modify: `src/llm_reward/models/__init__.py` — append `from . import lstm_baseline  # noqa: F401`
- Test: `tests/models/test_lstm_baseline.py`

**Interfaces:**
- Consumes: `PairwiseExample`, `require`, `LSTMConfig`, `ModelBundle`, `register`.
- Produces: `build_model(config: LSTMConfig) -> ModelBundle`, registered under `"lstm_baseline"`.

**Design note**: no pretrained vocabulary exists for a from-scratch baseline, so tokenization uses
a feature-hashing trick (`hash(token) % (vocab_size - 1) + 1`, 0 reserved for padding) instead of a
fitted vocabulary — this keeps `build_model` fully self-contained (no corpus-dependent state to
fit before training starts), matching "each model variant's tests build a real, tiny instance —
never a downloaded checkpoint."

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_lstm_baseline.py
from pathlib import Path

import torch

from llm_reward.models import lstm_baseline
from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle, build_model


def _config(**overrides):
    kwargs = dict(
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

    loss_before = torch.nn.functional.cross_entropy(
        bundle.model(batch["token_ids"]), batch["labels"]
    )
    optimizer.zero_grad()
    loss_before.backward()
    optimizer.step()
    with torch.no_grad():
        loss_after = torch.nn.functional.cross_entropy(
            bundle.model(batch["token_ids"]), batch["labels"]
        )
    assert loss_after.item() < loss_before.item()


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_lstm_baseline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.models.lstm_baseline'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/models/lstm_baseline.py
from __future__ import annotations

import torch
from torch import nn

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import LSTMConfig
from .registry import ModelBundle, register


def _format_input(example: PairwiseExample) -> str:
    return f"{example.prompt}\n[RESPONSE A]\n{example.response_a}\n[RESPONSE B]\n{example.response_b}"


def _hash_tokenize(text: str, vocab_size: int, max_seq_len: int) -> list[int]:
    tokens = text.split()[:max_seq_len]
    ids = [hash(tok) % (vocab_size - 1) + 1 for tok in tokens]  # 0 reserved for padding
    ids += [0] * (max_seq_len - len(ids))
    return ids


def make_collate_fn(vocab_size: int, max_seq_len: int):
    def collate_fn(batch: list[PairwiseExample]) -> dict[str, torch.Tensor]:
        require(len(batch) > 0, "collate_fn received an empty batch")
        token_ids = torch.tensor(
            [_hash_tokenize(_format_input(ex), vocab_size, max_seq_len) for ex in batch],
            dtype=torch.long,
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {"token_ids": token_ids, "labels": labels}

    return collate_fn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        _, (hidden, _) = self.lstm(embedded)
        return self.classifier(hidden[-1])


@register(LSTMConfig.variant)
def build_model(config: LSTMConfig) -> ModelBundle:
    model = LSTMClassifier(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    return ModelBundle(model=model, collate_fn=make_collate_fn(config.vocab_size, config.max_seq_len))
```

```yaml
# configs/lstm_baseline.yaml
variant: lstm_baseline
seed: 20260907
batch_size: 32
epochs: 5
lr: 0.001
output_dir: outputs/lstm_baseline
run_name: lstm-baseline-v1
max_seq_len: 256
vocab_size: 30000
embedding_dim: 256
hidden_dim: 512
num_layers: 2
```

```python
# append one line to src/llm_reward/models/__init__.py
from . import lstm_baseline  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_lstm_baseline.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/models/lstm_baseline.py src/llm_reward/models/__init__.py \
        configs/lstm_baseline.yaml tests/models/test_lstm_baseline.py
git commit -m "feat: add LSTM baseline model (hashing tokenizer, registered as lstm_baseline)"
```

---

## Track C — Small SFT + Head (depends only on Track 0)

### Task 10: `models/sft_head.py`

**Files:**
- Create: `src/llm_reward/models/sft_head.py`
- Create: `configs/small_sft_head.yaml`
- Modify: `src/llm_reward/models/__init__.py` — append `from . import sft_head  # noqa: F401`
- Test: `tests/models/test_sft_head.py`
- Test (integration, needs network): `tests/integration/test_sft_head_integration.py`

**Interfaces:**
- Consumes: `PairwiseExample`, `require`, `SFTHeadConfig`, `ModelBundle`, `register`.
- Produces: `build_model(config: SFTHeadConfig) -> ModelBundle`, registered under `"small_sft_head"`.

**Design notes (both verified locally against the installed `transformers`/`peft` versions
before writing this task):**
- `Qwen2ForSequenceClassification.forward()` raises `ValueError: Cannot handle batch sizes > 1
  if no padding token is defined` unless `model.config.pad_token_id` is set — Qwen models often
  ship without a dedicated pad token, so fall back to the tokenizer's EOS token when none exists.
- `forward()` returns a `SequenceClassifierOutputWithPast`, not a bare tensor — `_LogitsOnly`
  (below) unwraps `.logits`, per the spec's Section B amendment.
- The base (non-LoRA) classification head's parameters are named `score.weight` (no bias) —
  confirmed by inspecting `named_parameters()` on a real, tiny (from-config) instance — so
  `head_lr`'s param-group split checks `name.startswith("score")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_sft_head.py
from pathlib import Path

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


def _config(tmp_path, **overrides) -> SFTHeadConfig:
    kwargs = dict(
        seed=1, batch_size=2, epochs=1, lr=1e-4, output_dir=tmp_path, run_name="r",
        hf_model_name="unused-because-mocked", max_seq_len=16,
    )
    kwargs.update(overrides)
    return SFTHeadConfig(**kwargs)


def test_build_model_returns_working_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path))

    assert isinstance(bundle, ModelBundle)
    input_ids = torch.randint(1, 1000, (2, 8))
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    logits = bundle.model(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_one_training_step_lowers_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

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
    monkeypatch.setattr(sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path, head_lr=1e-2))
    assert bundle.param_groups is not None
    assert len(bundle.param_groups) == 2
    assert bundle.param_groups[1]["lr"] == 1e-2


def test_head_lr_none_leaves_param_groups_unset(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path))
    assert bundle.param_groups is None


def test_gradient_checkpointing_enables_input_require_grads_when_backbone_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path, gradient_checkpointing=True, freeze_backbone=True))
    assert bundle.model.hf_model.is_gradient_checkpointing


def test_real_model_is_reachable_through_the_wrapper_for_push_to_hub_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(sft_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = sft_head.build_model(_config(tmp_path))
    from transformers import PreTrainedModel

    assert isinstance(bundle.model.hf_model, PreTrainedModel)
```

```python
# tests/integration/test_sft_head_integration.py
import pytest
import torch

from llm_reward.models.config import SFTHeadConfig
from llm_reward.models.sft_head import build_model


@pytest.mark.integration
def test_real_tokenizer_produces_expected_batch_keys(tmp_path, tiny_pairwise_examples):
    """Downloads the real Qwen/Qwen2.5-0.5B-Instruct tokenizer on first run (cached after)."""
    config = SFTHeadConfig(
        seed=1, batch_size=2, epochs=1, lr=1e-4, output_dir=tmp_path, run_name="r",
        hf_model_name="Qwen/Qwen2.5-0.5B-Instruct", max_seq_len=64,
    )
    bundle = build_model(config)
    batch = bundle.collate_fn(tiny_pairwise_examples[:2])
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape[0] == 2
    assert isinstance(batch["labels"], torch.Tensor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_sft_head.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.models.sft_head'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/models/sft_head.py
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import SFTHeadConfig
from .registry import ModelBundle, register


def _format_input(example: PairwiseExample) -> str:
    return f"{example.prompt}\n[RESPONSE A]\n{example.response_a}\n[RESPONSE B]\n{example.response_b}"


class _LogitsOnly(nn.Module):
    """Unwraps a HF ModelOutput so forward(**inputs) returns a bare logits tensor, matching the
    ModelBundle contract instead of transformers' wrapper object."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.hf_model(**inputs).logits


def make_collate_fn(tokenizer, max_seq_len: int):
    def collate_fn(batch: list[PairwiseExample]) -> dict[str, torch.Tensor]:
        require(len(batch) > 0, "collate_fn received an empty batch")
        texts = [_format_input(ex) for ex in batch]
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    return collate_fn


@register(SFTHeadConfig.variant)
def build_model(config: SFTHeadConfig) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForSequenceClassification.from_pretrained(config.hf_model_name, num_labels=3)
    hf_model.config.pad_token_id = tokenizer.pad_token_id

    if config.freeze_backbone:
        for name, param in hf_model.named_parameters():
            if not name.startswith("score"):
                param.requires_grad = False

    if config.gradient_checkpointing:
        hf_model.gradient_checkpointing_enable()
        if config.freeze_backbone:
            # Without this, gradient checkpointing recomputes the frozen embedding layer's
            # forward pass with no grad-tracking input, and backward silently produces no
            # gradient for anything downstream. Only needed when something upstream is frozen.
            hf_model.enable_input_require_grads()

    param_groups = None
    if config.head_lr is not None:
        head_params = [p for n, p in hf_model.named_parameters() if p.requires_grad and n.startswith("score")]
        backbone_params = [
            p for n, p in hf_model.named_parameters() if p.requires_grad and not n.startswith("score")
        ]
        param_groups = [{"params": backbone_params}, {"params": head_params, "lr": config.head_lr}]

    return ModelBundle(
        model=_LogitsOnly(hf_model),
        collate_fn=make_collate_fn(tokenizer, config.max_seq_len),
        param_groups=param_groups,
    )
```

```yaml
# configs/small_sft_head.yaml
variant: small_sft_head
seed: 20260907
batch_size: 16
epochs: 3
lr: 0.00002
output_dir: outputs/small_sft_head
run_name: small-sft-head-v1
hf_model_name: Qwen/Qwen2.5-0.5B-Instruct
max_seq_len: 1024
head_lr: 0.001
mixed_precision: bf16
```

```python
# append one line to src/llm_reward/models/__init__.py
from . import sft_head  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_sft_head.py -v`
Expected: PASS (7 tests). The integration test needs network on first run:
`uv run pytest -m integration tests/integration/test_sft_head_integration.py -v`.

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/models/sft_head.py src/llm_reward/models/__init__.py \
        configs/small_sft_head.yaml tests/models/test_sft_head.py tests/integration/test_sft_head_integration.py
git commit -m "feat: add small SFT + classification head model (Qwen2.5-0.5B-Instruct)"
```

---

## Track D — Medium SFT + LoRA (depends only on Track 0)

### Task 11: `models/lora_head.py`

**Files:**
- Create: `src/llm_reward/models/lora_head.py`
- Create: `configs/medium_lora.yaml`
- Modify: `src/llm_reward/models/__init__.py` — append `from . import lora_head  # noqa: F401`
- Test: `tests/models/test_lora_head.py`
- Test (integration, needs network): `tests/integration/test_lora_head_integration.py`

**Interfaces:**
- Consumes: `PairwiseExample`, `require`, `LoRAConfig`, `ModelBundle`, `register`.
- Produces: `build_model(config: LoRAConfig) -> ModelBundle`, registered under `"medium_lora"`.

**Design note (verified locally against a real tiny Qwen2 + peft `LoraConfig(task_type="SEQ_CLS")`
instance before writing this task):** peft renames the classification head's parameter to
`...score.modules_to_save.default.weight` (it adds the head to `modules_to_save` automatically
under `task_type="SEQ_CLS"`, keeping it fully trainable while LoRA-wrapping the attention
projections), while every LoRA adapter parameter's name contains `lora_` and never
`modules_to_save`. `head_lr`'s param-group split checks `"modules_to_save" in name`.

- [ ] **Step 1: Write the failing test**

```python
# tests/models/test_lora_head.py
from pathlib import Path

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
    monkeypatch.setattr(lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
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
    monkeypatch.setattr(lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path))
    trainable = [n for n, p in bundle.model.hf_model.named_parameters() if p.requires_grad]
    assert trainable  # something is trainable
    assert all("lora_" in n or "modules_to_save" in n for n in trainable)


def test_one_training_step_lowers_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
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
    monkeypatch.setattr(lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path, head_lr=1e-2))
    assert bundle.param_groups is not None
    assert len(bundle.param_groups) == 2
    assert bundle.param_groups[1]["lr"] == 1e-2


def test_gradient_checkpointing_enables_input_require_grads(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path, gradient_checkpointing=True))
    assert bundle.model.hf_model.is_gradient_checkpointing


def test_real_model_is_reachable_through_the_wrapper_for_push_to_hub_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_head.AutoModelForSequenceClassification, "from_pretrained", _tiny_qwen2_model)
    monkeypatch.setattr(lora_head.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())

    bundle = lora_head.build_model(_config(tmp_path))
    from peft import PeftModel

    assert isinstance(bundle.model.hf_model, PeftModel)
```

```python
# tests/integration/test_lora_head_integration.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_lora_head.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.models.lora_head'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/models/lora_head.py
from __future__ import annotations

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data.pairwise import PairwiseExample
from ..negative_space import require
from .config import LoRAConfig
from .registry import ModelBundle, register


def _format_input(example: PairwiseExample) -> str:
    return f"{example.prompt}\n[RESPONSE A]\n{example.response_a}\n[RESPONSE B]\n{example.response_b}"


class _LogitsOnly(nn.Module):
    """Unwraps a HF ModelOutput so forward(**inputs) returns a bare logits tensor, matching the
    ModelBundle contract instead of transformers'/peft's wrapper object. Duplicated from
    sft_head.py rather than shared, per the spec: registry.py stays stable, this file stays
    independent of the sft_head track."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.hf_model(**inputs).logits


def make_collate_fn(tokenizer, max_seq_len: int):
    def collate_fn(batch: list[PairwiseExample]) -> dict[str, torch.Tensor]:
        require(len(batch) > 0, "collate_fn received an empty batch")
        texts = [_format_input(ex) for ex in batch]
        encoded = tokenizer(
            texts, padding=True, truncation=True, max_length=max_seq_len, return_tensors="pt"
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

    return collate_fn


@register(LoRAConfig.variant)
def build_model(config: LoRAConfig) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(config.hf_model_name, num_labels=3)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = PeftLoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        task_type="SEQ_CLS",
    )
    peft_model = get_peft_model(base_model, lora_config)

    if config.gradient_checkpointing:
        peft_model.gradient_checkpointing_enable()
        # LoRA always freezes the base, so this is always needed here (unlike sft_head.py, where
        # it's conditional on freeze_backbone) — without it, gradient checkpointing recomputes the
        # frozen embedding layer's forward pass with no grad-tracking input, and backward silently
        # produces no gradient for the adapters at all.
        peft_model.enable_input_require_grads()

    param_groups = None
    if config.head_lr is not None:
        head_params = [
            p for n, p in peft_model.named_parameters() if p.requires_grad and "modules_to_save" in n
        ]
        adapter_params = [
            p for n, p in peft_model.named_parameters() if p.requires_grad and "modules_to_save" not in n
        ]
        param_groups = [{"params": adapter_params}, {"params": head_params, "lr": config.head_lr}]

    return ModelBundle(
        model=_LogitsOnly(peft_model),
        collate_fn=make_collate_fn(tokenizer, config.max_seq_len),
        param_groups=param_groups,
    )
```

```yaml
# configs/medium_lora.yaml
variant: medium_lora
seed: 20260907
batch_size: 8
epochs: 3
lr: 0.0002
output_dir: outputs/medium_lora
run_name: medium-lora-v1
hf_model_name: Qwen/Qwen2.5-7B-Instruct
max_seq_len: 2048
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
head_lr: 0.001
mixed_precision: bf16
gradient_checkpointing: true
```

```python
# append one line to src/llm_reward/models/__init__.py
from . import lora_head  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_lora_head.py -v`
Expected: PASS (7 tests). Integration test needs network on first run.

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/models/lora_head.py src/llm_reward/models/__init__.py \
        configs/medium_lora.yaml tests/models/test_lora_head.py tests/integration/test_lora_head_integration.py
git commit -m "feat: add medium SFT + LoRA model (Qwen2.5-7B-Instruct)"
```

---

## Track E — Train/Eval Loop (depends only on Track 0 — does NOT wait on Tracks B/C/D)

### Task 12: `evaluate.py`

**Files:**
- Create: `src/llm_reward/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `require`.
- Produces: `EvalMetrics(loss, accuracy, n_examples, confusion_matrix, per_class_precision,
  per_class_recall, per_class_f1, macro_f1)`, `evaluate(model, loader, device) -> EvalMetrics`.

**Design note**: every classification metric below is derived from one 3x3 confusion-matrix count
(`confusion[true_label][predicted_label]`), never accumulated separately — one place a counting
bug could hide, not five. `device` is threaded through (not resolved here) because `train.py`
(Task 15) resolves it once and moves the model there; `evaluate()` only needs to move each batch
to match.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from llm_reward.evaluate import EvalMetrics, evaluate
from llm_reward.negative_space import CheckFailed

CPU = torch.device("cpu")


class _FixedBatchDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


class _IdentityOnFeatures(nn.Module):
    """Returns the pre-baked 'features' batch as-is. NOT nn.Identity(): its forward's parameter
    is named 'input', so calling it as model(features=...) — the actual ModelBundle contract,
    forward(**inputs) — raises TypeError. This class's parameter is named to match."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features


def _loader(logits_per_example: list[list[float]], labels: list[int]) -> DataLoader:
    """A DataLoader whose collate_fn hands back pre-baked logits (as 'features') and labels, so
    tests assert exact confusion-matrix-derived metrics without training anything."""

    def collate_fn(indices):
        features = torch.tensor([logits_per_example[i] for i in indices], dtype=torch.float32)
        batch_labels = torch.tensor([labels[i] for i in indices], dtype=torch.long)
        return {"features": features, "labels": batch_labels}

    return DataLoader(_FixedBatchDataset(len(labels)), batch_size=2, collate_fn=collate_fn)


def _one_hot_logit(predicted_class: int) -> list[float]:
    logit = [0.0, 0.0, 0.0]
    logit[predicted_class] = 10.0
    return logit


def test_evaluate_reports_perfect_accuracy_for_a_perfect_classifier():
    labels = [0, 1, 2, 0]
    loader = _loader([_one_hot_logit(label) for label in labels], labels)
    metrics = evaluate(_IdentityOnFeatures(), loader, CPU)
    assert isinstance(metrics, EvalMetrics)
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.n_examples == 4
    assert metrics.macro_f1 == pytest.approx(1.0)


def test_evaluate_computes_confusion_matrix_and_per_class_metrics_for_imperfect_predictions():
    # true labels: 0, 0, 1, 1, 2, 2 -- predicted: 0, 1, 1, 1, 2, 0
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 0]
    loader = _loader([_one_hot_logit(p) for p in predictions], labels)

    metrics = evaluate(_IdentityOnFeatures(), loader, CPU)

    assert metrics.confusion_matrix == ((1, 1, 0), (0, 2, 0), (1, 0, 1))
    assert metrics.accuracy == pytest.approx(4 / 6)
    assert metrics.per_class_precision == pytest.approx((0.5, 2 / 3, 1.0))
    assert metrics.per_class_recall == pytest.approx((0.5, 1.0, 0.5))
    assert metrics.per_class_f1 == pytest.approx((0.5, 0.8, 2 / 3))
    assert metrics.macro_f1 == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_evaluate_guards_against_a_class_absent_from_true_and_predicted_labels():
    # class 2 never appears as a true label or a prediction -- row/column sums are both zero.
    labels = [0, 1]
    predictions = [0, 1]
    loader = _loader([_one_hot_logit(p) for p in predictions], labels)

    metrics = evaluate(_IdentityOnFeatures(), loader, CPU)

    assert metrics.per_class_precision[2] == 0.0
    assert metrics.per_class_recall[2] == 0.0
    assert metrics.per_class_f1[2] == 0.0


def test_evaluate_rejects_an_empty_loader():
    loader = DataLoader(_FixedBatchDataset(0), batch_size=2, collate_fn=lambda x: x)
    with pytest.raises(CheckFailed, match="empty"):
        evaluate(_IdentityOnFeatures(), loader, CPU)


def test_evaluate_puts_the_model_in_eval_mode():
    model = _DropoutOnFeatures(p=0.9)  # in train() mode this would zero ~90% of activations
    loader = _loader([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [0, 1])
    evaluate(model, loader, CPU)
    assert not model.training
```

`_DropoutOnFeatures` is the same fix as `_IdentityOnFeatures` above, applied to `nn.Dropout`: its
`forward`'s parameter is named `input`, so `model(features=...)` — the real `evaluate()` contract
— raises `TypeError` without this wrapper. Add it next to `_IdentityOnFeatures`:

```python
class _DropoutOnFeatures(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dropout(features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.evaluate'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/evaluate.py
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .negative_space import require


@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    n_examples: int
    confusion_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    per_class_precision: tuple[float, float, float]
    per_class_recall: tuple[float, float, float]
    per_class_f1: tuple[float, float, float]
    macro_f1: float


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalMetrics:
    require(len(loader) > 0, "evaluate() received an empty DataLoader")
    model.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            logits = model(**inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
            total_loss += loss.item()
            total_examples += labels.shape[0]
            for true_label, pred_label in zip(labels.tolist(), logits.argmax(dim=-1).tolist()):
                confusion[true_label][pred_label] += 1
    require(total_examples > 0, "evaluate() processed zero examples")

    correct = sum(confusion[c][c] for c in range(3))
    precision, recall, f1 = [], [], []
    for c in range(3):
        true_positive = confusion[c][c]
        predicted_count = sum(confusion[i][c] for i in range(3))
        actual_count = sum(confusion[c])
        p = true_positive / predicted_count if predicted_count > 0 else 0.0
        r = true_positive / actual_count if actual_count > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)

    return EvalMetrics(
        loss=total_loss / total_examples,
        accuracy=correct / total_examples,
        n_examples=total_examples,
        confusion_matrix=(tuple(confusion[0]), tuple(confusion[1]), tuple(confusion[2])),
        per_class_precision=tuple(precision),
        per_class_recall=tuple(recall),
        per_class_f1=tuple(f1),
        macro_f1=sum(f1) / 3,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluate() with confusion-matrix-derived per-class precision/recall/F1"
```

---

### Task 13: `train.py` — optimizer/scheduler construction

**Files:**
- Create: `src/llm_reward/train.py` (this task writes only `_make_optimizer`/`_make_scheduler`;
  later tasks in this track append to the same file)
- Test: `tests/test_train.py` (this task writes only the optimizer/scheduler tests; later tasks
  append more tests to the same file)

**Interfaces:**
- Consumes: `ModelBundle`, `TrainConfig`.
- Produces: `_make_optimizer(bundle, config) -> AdamW`, `_make_scheduler(optimizer, config, num_training_steps)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
from pathlib import Path

import torch
from torch import nn

from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle
from llm_reward.train import _make_optimizer, _make_scheduler


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def _lstm_config(tmp_path, **overrides) -> LSTMConfig:
    kwargs = dict(seed=1, batch_size=2, epochs=2, lr=1e-2, output_dir=tmp_path, run_name="r")
    kwargs.update(overrides)
    return LSTMConfig(**kwargs)


def test_make_optimizer_uses_flat_param_list_when_no_param_groups(tmp_path):
    model = _TinyClassifier()
    bundle = ModelBundle(model=model, collate_fn=lambda batch: {})
    optimizer = _make_optimizer(bundle, _lstm_config(tmp_path, lr=0.05))
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.05


def test_make_optimizer_respects_bundle_param_groups(tmp_path):
    model = _TinyClassifier()
    param_groups = [
        {"params": [model.linear.weight]},
        {"params": [model.linear.bias], "lr": 0.5},
    ]
    bundle = ModelBundle(model=model, collate_fn=lambda batch: {}, param_groups=param_groups)
    optimizer = _make_optimizer(bundle, _lstm_config(tmp_path, lr=0.01))

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 0.01  # inherits config.lr as the default
    assert optimizer.param_groups[1]["lr"] == 0.5  # keeps its own override


def test_make_scheduler_runs_without_a_model_specific_argument(tmp_path):
    model = _TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = _make_scheduler(optimizer, _lstm_config(tmp_path, warmup_ratio=0.1), num_training_steps=10)
    assert scheduler is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm_reward.train'`

- [ ] **Step 3: Write the implementation**

```python
# src/llm_reward/train.py
from __future__ import annotations

from torch.optim import AdamW
from transformers import get_scheduler

from .models.config import TrainConfig
from .models.registry import ModelBundle


def _make_optimizer(bundle: ModelBundle, config: TrainConfig) -> AdamW:
    params = (
        bundle.param_groups
        if bundle.param_groups is not None
        else [p for p in bundle.model.parameters() if p.requires_grad]
    )
    return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)


def _make_scheduler(optimizer: AdamW, config: TrainConfig, num_training_steps: int):
    return get_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(config.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/train.py tests/test_train.py
git commit -m "feat: add train.py optimizer/scheduler construction with param_groups fallback"
```

---

### Task 14: `train.py` — checkpoint save (disk-bounded)

**Files:**
- Modify: `src/llm_reward/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: `Checkpoint`, `_make_optimizer`/`_make_scheduler` (Task 13).
- Produces: `_last_path(config)`, `_best_path(config)`, `_save_checkpoint(path, checkpoint)`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_train.py
import torch

from llm_reward.models.checkpoint import Checkpoint
from llm_reward.train import _best_path, _last_path, _save_checkpoint


def test_last_and_best_paths_are_under_output_dir(tmp_path):
    config = _lstm_config(tmp_path)
    assert _last_path(config) == tmp_path / "last.pt"
    assert _best_path(config) == tmp_path / "best.pt"


def test_save_checkpoint_creates_parent_directories(tmp_path):
    config = _lstm_config(tmp_path / "nested" / "dir")
    checkpoint = Checkpoint(
        epoch=0, global_step=0, model_state={}, optimizer_state={}, scheduler_state=None,
        best_val_metric=0.0, config=config, wandb_run_id="r",
    )
    path = _last_path(config)
    _save_checkpoint(path, checkpoint)
    assert path.exists()
    loaded: Checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    assert loaded.epoch == 0


def test_exactly_two_checkpoint_files_exist_regardless_of_epoch_count(tmp_path):
    """Bounded disk use: only last.pt and best.pt, never one file per epoch."""
    config = _lstm_config(tmp_path)
    for epoch in range(5):
        checkpoint = Checkpoint(
            epoch=epoch, global_step=epoch * 4, model_state={}, optimizer_state={}, scheduler_state=None,
            best_val_metric=float(epoch), config=config, wandb_run_id="r",
        )
        _save_checkpoint(_last_path(config), checkpoint)
        _save_checkpoint(_best_path(config), checkpoint)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt", "last.pt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ImportError: cannot import name '_last_path'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/llm_reward/train.py
from pathlib import Path

import torch

from .models.checkpoint import Checkpoint


def _last_path(config: TrainConfig) -> Path:
    return Path(config.output_dir) / "last.pt"


def _best_path(config: TrainConfig) -> Path:
    return Path(config.output_dir) / "best.pt"


def _save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/train.py tests/test_train.py
git commit -m "feat: add train.py checkpoint saving, bounded to last.pt + best.pt"
```

---

### Task 15: `train.py` — the training loop, resume, and W&B

**Files:**
- Modify: `src/llm_reward/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- Consumes: everything from Tasks 12-14, plus `PairwiseDataset`, `ConfigMismatchError`.
- Produces: `train(config, bundle, train_examples, val_examples, *, resume: bool) -> Checkpoint`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_train.py
import dataclasses

import pytest

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.models.checkpoint import ConfigMismatchError
from llm_reward.negative_space import CheckFailed
from llm_reward.train import train


class _FakeRun:
    """Stands in for the object wandb.init(...) returns. train.py holds this object and calls
    .log(...)/.id/.define_metric(...) on it, and (if used as a context manager) .__enter__/
    .__exit__ -- never module-level wandb.log(...), which targets an implicit global run."""

    def __init__(self, run_id: str, logged: list[dict]) -> None:
        self.id = run_id
        self._logged = logged

    def log(self, data: dict) -> None:
        self._logged.append(data)

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.run: _FakeRun | None = None
        self._next_id = 0

    def init(self, *, project, config=None, id=None, resume=None):
        self.run = _FakeRun(id or f"fake-run-{self._next_id}", self.logged)
        self._next_id += 1
        return self.run

    def Table(self, *, columns, data):
        return {"columns": columns, "data": data}


def _fake_bundle(with_param_groups: bool = False):
    from torch import nn

    from llm_reward.models.registry import ModelBundle

    model = _TinyClassifier()

    def collate_fn(batch: list[PairwiseExample]):
        import torch

        features = torch.randn(len(batch), 4)
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
        return {"features": features, "labels": labels}

    param_groups = None
    if with_param_groups:
        param_groups = [
            {"params": [model.linear.weight]},
            {"params": [model.linear.bias], "lr": 0.5},
        ]
    return ModelBundle(model=model, collate_fn=collate_fn, param_groups=param_groups)


def _examples(n: int) -> list[PairwiseExample]:
    return [
        PairwiseExample(id=str(i), prompt="p", response_a="a", response_b="b", label=i % 3)
        for i in range(n)
    ]


def _epoch_entries(logged: list[dict]) -> list[dict]:
    """Per-epoch summary entries (they carry an 'epoch' key); per-step entries don't."""
    return [entry for entry in logged if "epoch" in entry]


def test_train_runs_and_writes_both_checkpoint_files(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=2, batch_size=2)
    bundle = _fake_bundle()

    train(config, bundle, _examples(8), _examples(4), resume=False)

    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()


def test_train_moves_model_and_batches_to_the_resolved_device(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)
    bundle = _fake_bundle()

    train(config, bundle, _examples(8), _examples(4), resume=False)

    # These tests run on CPU only (no GPU in CI); this asserts the model actually landed on the
    # device train() resolved, not merely that training didn't crash.
    assert next(bundle.model.parameters()).device.type == "cpu"


def test_train_logs_per_step_loss_lr_and_grad_norm(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    step_entries = [entry for entry in fake_wandb.logged if "train/global_step" in entry]
    assert len(step_entries) == 4  # 8 train examples / batch_size 2
    assert {"train/loss", "train/lr", "train/grad_norm", "train/global_step"} <= step_entries[0].keys()
    assert [entry["train/global_step"] for entry in step_entries] == [0, 1, 2, 3]


def test_train_logs_split_grad_norm_when_bundle_has_param_groups(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(with_param_groups=True), _examples(8), _examples(4), resume=False)

    step_entries = [entry for entry in fake_wandb.logged if "train/global_step" in entry]
    assert "train/grad_norm_backbone" in step_entries[0]
    assert "train/grad_norm_head" in step_entries[0]
    assert "train/grad_norm" not in step_entries[0]


def test_train_logs_epoch_summary_with_full_metric_set(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=2, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    epoch_entries = _epoch_entries(fake_wandb.logged)
    assert [entry["epoch"] for entry in epoch_entries] == [0, 1]
    assert {
        "train/epoch_loss", "train/epoch_accuracy", "val/loss", "val/accuracy", "val/macro_f1",
        "val/precision_a", "val/recall_a", "val/f1_a", "val/confusion_matrix",
    } <= epoch_entries[0].keys()


def test_resume_continues_from_the_next_epoch(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=2, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)
    assert [entry["epoch"] for entry in _epoch_entries(fake_wandb.logged)] == [0, 1]

    resumed_config = _lstm_config(tmp_path, epochs=4, batch_size=2)
    fake_wandb.logged.clear()
    train(resumed_config, _fake_bundle(), _examples(8), _examples(4), resume=True)
    assert [entry["epoch"] for entry in _epoch_entries(fake_wandb.logged)] == [2, 3]


def test_resume_continues_the_global_step_counter(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)  # 4 steps: 0..3

    resumed_config = _lstm_config(tmp_path, epochs=2, batch_size=2)
    fake_wandb.logged.clear()
    train(resumed_config, _fake_bundle(), _examples(8), _examples(4), resume=True)
    step_entries = [entry for entry in fake_wandb.logged if "train/global_step" in entry]
    assert [entry["train/global_step"] for entry in step_entries] == [4, 5, 6, 7]


def test_resume_with_a_wandb_run_id_reuses_it(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    fake_wandb = _FakeWandb()
    monkeypatch.setattr(train_module, "wandb", fake_wandb)
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)

    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)
    first_run_id = fake_wandb.run.id

    resumed_config = _lstm_config(tmp_path, epochs=2, batch_size=2)
    train(resumed_config, _fake_bundle(), _examples(8), _examples(4), resume=True)
    assert fake_wandb.run.id == first_run_id


def test_resume_with_a_changed_config_raises_config_mismatch_error(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)
    train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)

    different_config = _lstm_config(tmp_path, epochs=2, batch_size=4)  # batch_size changed
    with pytest.raises(ConfigMismatchError):
        train(different_config, _fake_bundle(), _examples(8), _examples(4), resume=True)


def test_train_rejects_zero_epochs(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=0, batch_size=2)
    with pytest.raises(CheckFailed, match="epochs"):
        train(config, _fake_bundle(), _examples(8), _examples(4), resume=False)


def test_train_rejects_empty_train_examples(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config = _lstm_config(tmp_path, epochs=1, batch_size=2)
    with pytest.raises(CheckFailed, match="training examples"):
        train(config, _fake_bundle(), [], _examples(4), resume=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ImportError: cannot import name 'train'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/llm_reward/train.py
import dataclasses

import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader

from .data.dataset import PairwiseDataset
from .evaluate import evaluate
from .models.checkpoint import ConfigMismatchError
from .negative_space import require


def _group_grad_norm(params) -> float:
    """The L2 norm of one param group's gradients, computed WITHOUT clipping them -- purely for
    observability. Must be called before the single combined clip_grad_norm_ call below, or it
    would report the post-clip norm instead."""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.norm(torch.stack([g.norm() for g in grads])).item()


def train(config: TrainConfig, bundle: ModelBundle, train_examples, val_examples, *, resume: bool) -> Checkpoint:
    require(config.epochs >= 1, f"epochs must be at least 1, got {config.epochs}")
    require(len(train_examples) > 0, "train() received zero training examples")
    require(len(val_examples) > 0, "train() received zero validation examples")

    device = torch.accelerator.current_accelerator(check_available=True) or torch.device("cpu")
    bundle.model.to(device)

    train_loader = DataLoader(
        PairwiseDataset(train_examples), batch_size=config.batch_size, shuffle=True,
        collate_fn=bundle.collate_fn,
    )
    val_loader = DataLoader(
        PairwiseDataset(val_examples), batch_size=config.batch_size, shuffle=False,
        collate_fn=bundle.collate_fn,
    )

    optimizer = _make_optimizer(bundle, config)
    scheduler = _make_scheduler(optimizer, config, len(train_loader) * config.epochs)
    class_weights = (
        torch.tensor(config.class_weights, dtype=torch.float32, device=device)
        if config.class_weights else None
    )

    last_path = _last_path(config)
    run_config = dataclasses.asdict(config) | {
        "variant": config.variant,
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
    }

    if resume and last_path.exists():
        checkpoint: Checkpoint = torch.load(last_path, weights_only=False, map_location=device)
        # epochs is deliberately excluded from the comparison: resuming specifically to train for
        # MORE epochs (test_resume_continues_from_the_next_epoch) is the whole point of --resume,
        # so a config that only differs in epochs must NOT raise here. Every other field changing
        # (e.g. batch_size, lr) still must, since those invalidate the saved optimizer/scheduler
        # state. Confirmed by running both resume tests: the epochs-only-differs case must pass,
        # the batch_size-differs case must still raise.
        if dataclasses.replace(checkpoint.config, epochs=config.epochs) != config:
            raise ConfigMismatchError(f"resume checkpoint at {last_path} was produced by a different config")
        bundle.model.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        if checkpoint.scheduler_state is not None:
            scheduler.load_state_dict(checkpoint.scheduler_state)
        start_epoch = checkpoint.epoch + 1
        global_step = checkpoint.global_step
        best_val_metric = checkpoint.best_val_metric
        init_kwargs = {"id": checkpoint.wandb_run_id, "resume": "must"}
    else:
        start_epoch = 0
        global_step = 0
        best_val_metric = float("-inf")
        checkpoint = None
        init_kwargs = {"resume": "allow"}

    with wandb.init(project="llm-reward", config=run_config, **init_kwargs) as run:
        run.define_metric("train/global_step")
        run.define_metric("train/*", step_metric="train/global_step")
        run.define_metric("epoch")
        run.define_metric("val/*", step_metric="epoch")

        for epoch in range(start_epoch, config.epochs):
            bundle.model.train()
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_n = 0

            for batch in train_loader:
                require("labels" in batch, "collate_fn output is missing the required 'labels' key")
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                inputs = {k: v for k, v in batch.items() if k != "labels"}

                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=config.mixed_precision == "bf16"
                ):
                    logits = bundle.model(**inputs)
                    loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)

                optimizer.zero_grad()
                loss.backward()

                if bundle.param_groups is not None:
                    grad_norm_backbone = _group_grad_norm(bundle.param_groups[0]["params"])
                    grad_norm_head = _group_grad_norm(bundle.param_groups[1]["params"])
                grad_norm = nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()

                epoch_loss_sum += loss.item() * labels.shape[0]
                epoch_correct += (logits.argmax(dim=-1) == labels).sum().item()
                epoch_n += labels.shape[0]

                step_log = {
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                }
                if bundle.param_groups is not None:
                    step_log["train/grad_norm_backbone"] = grad_norm_backbone
                    step_log["train/grad_norm_head"] = grad_norm_head
                else:
                    step_log["train/grad_norm"] = float(grad_norm)
                run.log(step_log)
                global_step += 1

            metrics = evaluate(bundle.model, val_loader, device)
            epoch_log = {
                "epoch": epoch,
                "train/epoch_loss": epoch_loss_sum / epoch_n,
                "train/epoch_accuracy": epoch_correct / epoch_n,
                "val/loss": metrics.loss,
                "val/accuracy": metrics.accuracy,
                "val/macro_f1": metrics.macro_f1,
                "val/precision_a": metrics.per_class_precision[0],
                "val/precision_b": metrics.per_class_precision[1],
                "val/precision_tie": metrics.per_class_precision[2],
                "val/recall_a": metrics.per_class_recall[0],
                "val/recall_b": metrics.per_class_recall[1],
                "val/recall_tie": metrics.per_class_recall[2],
                "val/f1_a": metrics.per_class_f1[0],
                "val/f1_b": metrics.per_class_f1[1],
                "val/f1_tie": metrics.per_class_f1[2],
                "val/confusion_matrix": wandb.Table(
                    columns=["true_label", "pred_a", "pred_b", "pred_tie"],
                    data=[
                        [name, *row]
                        for name, row in zip(("a", "b", "tie"), metrics.confusion_matrix)
                    ],
                ),
            }
            # torch.accelerator has no memory-accounting API in this project's pinned torch==2.8.0
            # (confirmed empty: dir(torch.accelerator) has no "memory" member; added in a later
            # release) -- go through the device-specific module instead so this doesn't crash on
            # the RunPod CUDA target or on an Apple Silicon dev machine (MPS).
            if device.type == "cuda":
                epoch_log["system/gpu_mem_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(device) / 1e6
                )
            elif device.type == "mps":
                epoch_log["system/gpu_mem_allocated_mb"] = (
                    torch.mps.current_allocated_memory() / 1e6
                )
            run.log(epoch_log)

            improved = metrics.accuracy > best_val_metric
            best_val_metric = max(best_val_metric, metrics.accuracy)
            checkpoint = Checkpoint(
                epoch=epoch,
                global_step=global_step,
                model_state=bundle.model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                best_val_metric=best_val_metric,
                config=config,
                wandb_run_id=run.id,
            )
            _save_checkpoint(last_path, checkpoint)
            if improved:
                _save_checkpoint(_best_path(config), checkpoint)

    return checkpoint
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v`
Expected: PASS (17 tests)

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/train.py tests/test_train.py
git commit -m "feat: add train() with device placement, AMP, resume, and full W&B metrics"
```

---

### Task 16: `train.py` — CLI entrypoint

**Files:**
- Modify: `src/llm_reward/train.py`
- Modify: `pyproject.toml` — replace the `[project.scripts]` entry
- Test: `tests/test_train_cli.py`

**Interfaces:**
- Consumes: `load_config`, `build_model`, `load_pairwise_examples`, `split_train_val`, `train`.
- Produces: `main() -> None`, invocable as `uv run python -m llm_reward.train --config <path> [--resume]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_cli.py
from pathlib import Path

import pytest

from llm_reward.data.pairwise import PairwiseExample
from llm_reward.models.config import LSTMConfig
from llm_reward.models.registry import ModelBundle
from llm_reward.train import main


class _FakeRun:
    def __init__(self, logged: list[dict]) -> None:
        self.id = "fake"
        self._logged = logged

    def log(self, data: dict) -> None:
        self._logged.append(data)

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


class _FakeWandb:
    def __init__(self) -> None:
        self.logged: list[dict] = []

    def init(self, **kwargs):
        self.run = _FakeRun(self.logged)
        return self.run

    def Table(self, *, columns, data):
        return {"columns": columns, "data": data}


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    config_path.write_text(
        "variant: lstm_baseline\n"
        "seed: 1\nbatch_size: 2\nepochs: 1\nlr: 0.01\n"
        f"output_dir: {output_dir}\nrun_name: cli-test\n"
        "max_seq_len: 8\nvocab_size: 100\n"
    )
    return config_path


def _write_train_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "train.csv"
    lines = ["id,model_a,model_b,prompt,response_a,response_b,winner_model_a,winner_model_b,winner_model_tie"]
    for i in range(10):
        label_cols = ["0", "0", "0"]
        label_cols[i % 3] = "1"
        lines.append(f'{i},m,n,"prompt {i}","resp a {i}","resp b {i}",{",".join(label_cols)}')
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def test_main_runs_end_to_end_and_writes_checkpoints(tmp_path, monkeypatch):
    import llm_reward.train as train_module

    monkeypatch.setattr(train_module, "wandb", _FakeWandb())
    config_path = _write_config(tmp_path)
    csv_path = _write_train_csv(tmp_path)

    monkeypatch.setattr(
        "sys.argv",
        ["train", "--config", str(config_path), "--train-csv", str(csv_path)],
    )
    main()

    output_dir = tmp_path / "output"
    assert (output_dir / "last.pt").exists()
    assert (output_dir / "best.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'main'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/llm_reward/train.py
import argparse

from dotenv import load_dotenv

from .data.pairwise import load_pairwise_examples, split_train_val
from .models.config import load_config
from .models.registry import build_model


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train an llm-reward model variant")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = build_model(config)
    examples = load_pairwise_examples(args.train_csv)
    train_examples, val_examples = split_train_val(examples, config.val_fraction, config.seed)
    train(config, bundle, train_examples, val_examples, resume=args.resume)


if __name__ == "__main__":
    main()
```

```toml
# in pyproject.toml, replace:
[project.scripts]
llm-reward = "llm_reward:main"

# with:
[project.scripts]
llm-reward-train = "llm_reward.train:main"
```

Note: importing `models` (the `models/__init__.py` package) inside `build_model`'s call path
triggers the `@register` decorators in `lstm_baseline.py`/`sft_head.py`/`lora_head.py` — but only
once those tracks have appended their import lines to `models/__init__.py` (Tasks 9-11). Until
then, `models/__init__.py` is empty and `build_model` raises `unknown variant` for every config —
correct and expected mid-implementation, not a bug in this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train_cli.py -v`
Expected: PASS (1 test) — requires Task 9 (`lstm_baseline`) to already be merged, since this test
uses `variant: lstm_baseline`. If Track B hasn't landed yet when this task runs, this is the one
genuine cross-track dependency in the whole plan: this specific *test* needs one real registered
variant to exist, even though `train.py` itself does not. Coordinate with whoever owns Track B, or
temporarily register a trivial fake variant for this test alone if Track B is still in flight.

- [ ] **Step 5: Commit**

```bash
git add src/llm_reward/train.py pyproject.toml tests/test_train_cli.py
git commit -m "feat: add train.py CLI entrypoint (llm-reward-train script)"
```

---

## Track F — Model Persistence (depends only on Track 0)

### Task 17: `scripts/push_to_hub.py`

**Files:**
- Create: `scripts/push_to_hub.py`
- Test: `tests/scripts/test_push_to_hub.py`

**Interfaces:**
- Consumes: `Checkpoint`, `build_model`, `SFTHeadConfig`, `LoRAConfig`.
- Produces: `push_to_hub(checkpoint_path: Path, repo_id: str, private: bool = True) -> str`.

**Design note**: dispatch unwraps `_LogitsOnly` via `getattr(bundle.model, "hf_model", bundle.model)`
before the `isinstance` checks, per the spec's Section E update. The non-HF (LSTM) branch follows
Hugging Face's own documented pattern for a custom `push_to_hub` (save to a temp directory, then
`HfApi().upload_folder`) rather than a single `upload_file` call with raw bytes, since that pattern
is the one confirmed in `huggingface_hub`'s own integration guide.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_push_to_hub.py
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import nn

from llm_reward.models.checkpoint import Checkpoint
from llm_reward.models.config import TrainConfig
from llm_reward.models.registry import ModelBundle, register
from llm_reward.negative_space import CheckFailed
from scripts.push_to_hub import push_to_hub


class _FakeApi:
    def __init__(self) -> None:
        self.created_repos = []
        self.uploaded_folders = []

    def create_repo(self, repo_id, private=False, exist_ok=True):
        self.created_repos.append((repo_id, private))

    def upload_folder(self, repo_id, folder_path):
        self.uploaded_folders.append((repo_id, Path(folder_path)))
        return f"https://huggingface.co/{repo_id}"


def test_push_to_hub_uploads_a_plain_module_via_upload_folder(tmp_path, monkeypatch):
    @dataclass(frozen=True, kw_only=True)
    class _PushTestConfig(TrainConfig):
        variant: ClassVar[str] = "test_push_variant"

    @register("test_push_variant")
    def _build(config):
        return ModelBundle(model=nn.Linear(4, 3), collate_fn=lambda batch: {})

    config = _PushTestConfig(seed=1, batch_size=2, epochs=1, lr=1e-2, output_dir=tmp_path, run_name="r")
    checkpoint = Checkpoint(
        epoch=0, global_step=4, model_state=nn.Linear(4, 3).state_dict(), optimizer_state={},
        scheduler_state=None, best_val_metric=0.9, config=config, wandb_run_id="r",
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(checkpoint, checkpoint_path)

    fake_api = _FakeApi()
    import scripts.push_to_hub as push_module

    monkeypatch.setattr(push_module, "HfApi", lambda: fake_api)

    result = push_to_hub(checkpoint_path, repo_id="me/test-model", private=True)

    assert fake_api.created_repos == [("me/test-model", True)]
    assert len(fake_api.uploaded_folders) == 1
    repo_id, folder_path = fake_api.uploaded_folders[0]
    assert repo_id == "me/test-model"
    assert (folder_path / "model_state_dict.pt").exists()
    assert (folder_path / "README.md").exists()
    assert "0.9" in (folder_path / "README.md").read_text()
    assert result == "https://huggingface.co/me/test-model"


def test_push_to_hub_raises_on_missing_checkpoint(tmp_path):
    with pytest.raises(CheckFailed, match="not found"):
        push_to_hub(tmp_path / "does_not_exist.pt", repo_id="me/x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/scripts/test_push_to_hub.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/push_to_hub.py
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, PreTrainedModel

from llm_reward.models.checkpoint import Checkpoint
from llm_reward.models.config import LoRAConfig, SFTHeadConfig
from llm_reward.models.registry import build_model
from llm_reward.negative_space import require

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover — peft is a project dependency, always installed
    PeftModel = ()  # type: ignore[assignment]


def _model_card_text(checkpoint: Checkpoint) -> str:
    config_lines = "\n".join(f"- **{key}**: {value}" for key, value in vars(checkpoint.config).items())
    return (
        f"# {checkpoint.config.run_name}\n\n"
        f"Variant: `{checkpoint.config.variant}`\n\n"
        f"Best validation accuracy: {checkpoint.best_val_metric:.4f}\n\n"
        f"## Training configuration\n\n{config_lines}\n"
    )


def push_to_hub(checkpoint_path: Path, repo_id: str, private: bool = True) -> str:
    require(checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}")
    checkpoint: Checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    bundle = build_model(checkpoint.config)
    bundle.model.load_state_dict(checkpoint.model_state)
    real_model = getattr(bundle.model, "hf_model", bundle.model)

    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)

    if isinstance(real_model, PeftModel) or isinstance(real_model, PreTrainedModel):
        result = real_model.push_to_hub(repo_id, private=private)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            torch.save(real_model.state_dict(), tmp_path / "model_state_dict.pt")
            (tmp_path / "README.md").write_text(_model_card_text(checkpoint))
            result = api.upload_folder(repo_id=repo_id, folder_path=str(tmp_path))

    if isinstance(checkpoint.config, (SFTHeadConfig, LoRAConfig)):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint.config.hf_model_name)
        tokenizer.push_to_hub(repo_id, private=private)

    return str(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/scripts/test_push_to_hub.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/push_to_hub.py tests/scripts/test_push_to_hub.py
git commit -m "feat: add scripts/push_to_hub.py with capability-based push dispatch"
```

---

## Self-Review

**1. Spec coverage:**
- Section A (data contract) → Tasks 2, 7, 8.
- Section B (config + registry contract, including the `kw_only` fix and `_LogitsOnly`) → Tasks
  4, 5, 9, 10, 11.
- Section C (checkpoint/resume/W&B) → Tasks 6, 13, 14, 15, 16.
- Section D (parallelization) → the Track 0 / Track A-F structure of this whole document.
- Section E (Hub persistence) → Task 17.
- Negative-space/fail-fast rules from `CLAUDE.md` → Task 1's helper module, used throughout.
- pytest-expert conventions from `CLAUDE.md` → Task 3's conftest, the shared `tiny_pairwise_examples`
  fixture, and the integration-marked network-touching tests in Tasks 8, 10, 11.

**2. Placeholder scan:** none found — every step has real, complete code; the one genuine
open item (Task 7's note about verifying `_extract_text` against a real downloaded file) is
flagged explicitly as a verification step with concrete instructions, not a TBD.

**3. Type consistency:** `ModelBundle(model, collate_fn, param_groups)` is identical across
Tasks 5, 9, 10, 11, 13, 15, 17. `Checkpoint`'s eight fields (including `global_step`, added in this
revision) are identical across Tasks 6, 14, 15, 17. `EvalMetrics`'s eight fields (confusion matrix
plus per-class precision/recall/F1 plus macro-F1, also added in this revision) and `evaluate()`'s
three-argument signature (`model, loader, device`) are identical across Tasks 12 and 15.
`build_model`/`register`/`_REGISTRY` names are identical across Tasks 5, 9, 10, 11, 16.
`load_config`/`ConfigError` names are identical across Tasks 4 and 16.

**4. Post-pytorch-skill-review pass (this revision):** the plan was checked against the pytorch
skill (verified live against the installed `torch==2.8.0`) and the observability-expert skill
before any implementer was dispatched, per user request. Real findings and fixes, each propagated
through every task that constructs the affected type:
- Device placement was entirely absent from the original `train()`/`evaluate()` draft — a
  correctness bug (training would silently never touch the GPU), not a style issue. Fixed in
  Tasks 12 and 15; stated as a hard requirement in Global Constraints so it can't quietly regress.
- Also caught in the same pass: Task 12's original test called `nn.Identity()` with a `features=`
  kwarg, which raises `TypeError` (`Identity.forward`'s parameter is named `input`) — verified
  with a real interpreter before writing the fix (`_IdentityOnFeatures`, Task 12).
- Mixed precision (`mixed_precision` config field, Task 4), gradient checkpointing +
  `enable_input_require_grads` for the two HF-backed variants (Tasks 10-11, verified against
  peft's own parameter-naming behavior with a real tiny model, not assumed), and `map_location` on
  every `torch.load` of a `Checkpoint` (Tasks 6, 15, 17) were missing and are now present.
- W&B logging expanded from `{epoch, val_loss, val_accuracy}` to the full set described in Task 15
  — per-step loss/LR/grad-norm (split by param group when `head_lr` is set), per-epoch train/val
  loss and accuracy on the same `epoch` axis, and confusion-matrix-derived per-class
  precision/recall/F1/macro-F1 — reviewed against observability-expert's W&B rules (hold the `run`
  object, declare x-axes via `define_metric` rather than passing `step=`).
