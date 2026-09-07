# Training seams design: data / models / train-eval

Date: 2026-09-07
Status: approved pending final user sign-off (see "Open questions" — none blocking)

## Purpose

Three model variants (LSTM baseline, small SFT model + classification head, medium SFT model +
LoRA) share one data pipeline and one train/eval loop, per `CLAUDE.md`. The intent going forward is
to build these pieces with multiple Claude Code sessions running in parallel — one per component —
so this document fixes the interfaces between them *before* any of that work starts. Every type
signature below is a contract: changing one after parallel work begins means renegotiating with
whichever other session depends on it, which is exactly what this document exists to avoid.

Nothing in this repository has been implemented yet. This is pure interface design.

## Scope

In scope: the seam between `data/` and `models/` (tokenization ownership), the seam between
`models/` and `train.py`/`evaluate.py` (the model + config + registry contract), the
checkpoint/resume/W&B contract inside the training loop itself, and (added 2026-09-07) the
opt-in Hugging Face Hub upload step that persists a finished run beyond the RunPod pod's lifecycle.

Out of scope (deliberately, YAGNI): `submit.py`'s prediction interface (writing
`submission.csv`) is mentioned only where it touches `evaluate.py`; its own contract is a smaller,
later design once the training loop exists. Also out of scope: RunPod provisioning (already
decided in `CLAUDE.md` — manual, not scripted) and the `pytest-expert`/negative-space testing
conventions, which already apply per `CLAUDE.md` and are referenced but not re-specified here.

## Decisions

Five forks were resolved before writing this spec; each is load-bearing for what follows.

1. **Tokenization is per-variant, not shared.** `data/dataset.py` stays at the raw-text level.
   Each model variant owns its own tokenizer/vocab and supplies the `collate_fn` that turns text
   into tensors. Rejected: one shared tokenizer for all three variants, because it would force the
   from-scratch LSTM baseline to use an HF subword vocab, which is not what a baseline is for.
2. **The model contract is a plain `nn.Module`; loss lives in `train.py`.** All three variants
   produce the same output shape (`[B, 3]` logits) and use the same loss (cross-entropy over the
   3-way label), so there is nothing variant-specific about the loss. Rejected: a richer
   `typing.Protocol` with `compute_loss`/`parameters_to_optimize` methods — unneeded surface area
   for three variants with an identical loss and a loss-agnostic optimizer param filter
   (`filter(lambda p: p.requires_grad, model.parameters())` already handles LoRA's frozen backbone
   with no model-specific method).
3. **Resume is epoch-boundary, not step-level.** Checkpoints are written at the end of each epoch;
   resuming re-runs the current epoch from scratch. Justified by dataset size (55K rows) — even the
   medium LoRA variant's epoch is cheap enough that losing a partial one to an interruption is
   acceptable. Rejected: step-level resume with a resumable sampler — real complexity (deterministic
   dataloader replay, mid-epoch optimizer/scheduler state) not justified at this data scale.
4. **Checkpoint retention is last + best only**, not one file per epoch. `last.pt` is the resume
   point, overwritten every epoch; `best.pt` is the eval/submission point, overwritten only on
   improvement. Bounded disk use regardless of epoch count, which matters directly: the RunPod
   pod's `/workspace` is a 50GB persistent volume shared across all three variants' checkpoints, the
   HF model cache, and the Kaggle data (see `CLAUDE.md`'s RunPod section).
5. **Config is a base dataclass plus one frozen subclass per variant**, not one flat dataclass with
   every field optional. `LoRAConfig.lora_rank` cannot be set on an `LSTMConfig` because the type
   does not have that field — negative-space rule 9 (illegal states unrepresentable) applied to
   config loading, not just to data.

## A. Data contract

`data/pairwise.py`:

```python
@dataclass(frozen=True)
class PairwiseExample:
    id: str
    prompt: str
    response_a: str
    response_b: str
    label: int  # 0=a, 1=b, 2=tie — fixed encoding, asserted in __post_init__ (label in {0,1,2})

def load_pairwise_examples(csv_path: Path) -> list[PairwiseExample]:
    """Parses train.csv/test.csv into records. Owns tie handling and any prompt/response
    truncation *policy* (the truncation mechanics belong to each model's tokenizer, but the
    decision of what counts as too-long input text is a data-pipeline concern)."""

def split_train_val(
    examples: list[PairwiseExample], val_fraction: float, seed: int
) -> tuple[list[PairwiseExample], list[PairwiseExample]]:
    """Postconditions (negative-space, already implied by CLAUDE.md): disjoint id sets, and
    len(train) + len(val) == len(examples)."""
```

`data/dataset.py`:

```python
class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[PairwiseExample]): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> PairwiseExample: ...  # the raw dataclass, NEVER a tensor
```

**The seam**: `PairwiseDataset` never tokenizes. Tokenization happens only inside a model
variant's `collate_fn` (Section B), assembled into a `DataLoader` by `train.py`:

```python
DataLoader(PairwiseDataset(examples), batch_size=cfg.batch_size, shuffle=is_train,
           collate_fn=bundle.collate_fn)
```

This is what lets `data/` be built without knowing anything about tokenizers, and lets each model
variant be built without knowing anything about CSV parsing. The only thing both sides must agree
on is the five `PairwiseExample` field names and the label encoding.

## B. Model config + registry contract

`models/config.py` — one frozen base, one frozen subclass per variant:

```python
@dataclass(frozen=True)
class TrainConfig:
    variant: ClassVar[str]          # discriminator; overridden by each subclass
    seed: int
    batch_size: int
    epochs: int
    lr: float
    output_dir: Path
    run_name: str
    val_fraction: float = 0.1
    max_seq_len: int = 512

@dataclass(frozen=True)
class LSTMConfig(TrainConfig):
    variant: ClassVar[str] = "lstm_baseline"
    vocab_size: int = 30_000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2

@dataclass(frozen=True)
class SFTHeadConfig(TrainConfig):
    variant: ClassVar[str] = "small_sft_head"
    hf_model_name: str
    freeze_backbone: bool = False

@dataclass(frozen=True)
class LoRAConfig(TrainConfig):
    variant: ClassVar[str] = "medium_lora"
    hf_model_name: str
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

def load_config(yaml_path: Path) -> TrainConfig:
    """Reads the `variant:` key from the YAML file, picks the matching subclass, and constructs
    it — an unknown `variant:` value or a field that doesn't belong to the chosen subclass fails
    fast (typed exception: this is an operating error, a malformed config file, not a programmer
    error)."""
```

`models/registry.py`:

```python
CollateFn = Callable[[list[PairwiseExample]], dict[str, torch.Tensor]]

@dataclass(frozen=True)
class ModelBundle:
    model: nn.Module        # forward(**inputs) -> logits, shape [B, 3], dtype float32, finite
    collate_fn: CollateFn   # -> dict with at least "labels": LongTensor[B] in {0,1,2}

_REGISTRY: dict[str, Callable[[TrainConfig], ModelBundle]] = {}

def register(variant: str) -> Callable[[BuildFn], BuildFn]:
    def _decorator(fn: BuildFn) -> BuildFn:
        assert variant not in _REGISTRY, f"variant {variant!r} already registered"
        _REGISTRY[variant] = fn
        return fn
    return _decorator

def build_model(config: TrainConfig) -> ModelBundle:
    assert config.variant in _REGISTRY, f"unknown variant {config.variant!r}"
    return _REGISTRY[config.variant](config)
```

Each variant module registers itself:

```python
# models/lstm_baseline.py
@register(LSTMConfig.variant)
def build_model(config: LSTMConfig) -> ModelBundle: ...

# models/sft_head.py
@register(SFTHeadConfig.variant)
def build_model(config: SFTHeadConfig) -> ModelBundle: ...

# models/lora_head.py
@register(LoRAConfig.variant)
def build_model(config: LoRAConfig) -> ModelBundle: ...
```

`models/__init__.py` imports all three modules for their registration side effects:

```python
from . import lstm_baseline, sft_head, lora_head  # noqa: F401
```

**Why self-registration, not an if/elif in `registry.py`**: with a shared dispatch table, three
sessions adding three variants all edit the same lines in the same file — guaranteed merge
conflicts on exactly the file that is supposed to be the stable, shared seam. With `@register(...)`
declared in each variant's own file, `registry.py` never changes again after this design lands; the
only shared touch point is the three-line `models/__init__.py` import list, where three independent
one-line additions merge trivially.

**Uniform loss**, computed in `train.py`, never inside a model file:

```python
inputs = {k: v for k, v in batch.items() if k != "labels"}
loss = F.cross_entropy(bundle.model(**inputs), batch["labels"])
```

## C. Train/eval loop: checkpointing and resume

`models/checkpoint.py`:

```python
@dataclass
class Checkpoint:
    epoch: int                 # last COMPLETED epoch (0-indexed)
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict | None
    best_val_metric: float
    config: TrainConfig        # the exact config that produced this checkpoint
    wandb_run_id: str

class ConfigMismatchError(Exception):
    """Raised when --resume finds a checkpoint whose config != the config just loaded from YAML.
    Operating error (someone edited the YAML between runs), not a programmer error: raised and
    handled at the call site in train.py, never asserted."""
```

`evaluate.py`:

```python
@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    n_examples: int

def evaluate(model: nn.Module, loader: DataLoader) -> EvalMetrics: ...
```

`train.py` entry point:

```
uv run python -m llm_reward.train --config configs/<variant>.yaml [--resume]
```

Loop shape:

1. `config = load_config(args.config)`; `bundle = build_model(config)`.
2. Build train/val `DataLoader`s via `PairwiseDataset` + `bundle.collate_fn`.
3. If `--resume` and `{config.output_dir}/last.pt` exists:
   - Load the `Checkpoint`. `if checkpoint.config != config: raise ConfigMismatchError(...)`.
   - Restore `model_state`/`optimizer_state`/`scheduler_state`.
   - `start_epoch = checkpoint.epoch + 1`; `best_val_metric = checkpoint.best_val_metric`.
   - `wandb.init(id=checkpoint.wandb_run_id, resume="must", project=..., config=asdict(config))`.
4. Else: `start_epoch = 0`; `best_val_metric = -inf`; `wandb.init(resume="allow", ...)` with a
   freshly generated run id.
5. For `epoch in range(start_epoch, config.epochs)`: train one epoch, then
   `metrics = evaluate(bundle.model, val_loader)`, log to W&B, then:
   - always write `Checkpoint(epoch=epoch, ..., best_val_metric=max(best_val_metric, metrics.accuracy), wandb_run_id=wandb.run.id)` to `last.pt`.
   - if `metrics.accuracy > best_val_metric`: also write the same checkpoint to `best.pt` and
     update `best_val_metric`.

**Disk bound**: exactly two checkpoint files exist under `config.output_dir` at any time,
regardless of `config.epochs` — the "keep last + best only" decision from above, load-bearing given
the 50GB `/workspace` budget shared across all three variants.

## D. Parallelization plan

This is the reason the document exists, so it is worth stating explicitly rather than leaving it
implied by the interfaces:

1. **Shared-contract step (single-threaded, first)**: `models/config.py` (all four dataclasses),
   `models/registry.py` (`ModelBundle`, `register`, `build_model`), `models/checkpoint.py`
   (`Checkpoint`, `ConfigMismatchError`), and `data/pairwise.py`'s `PairwiseExample` dataclass. These
   are the types every other file imports; they should exist, reviewed, before any variant work
   starts. Small enough for one session (or the user) to write directly from this spec.
2. **Parallel from here**, each session touching disjoint files:
   - `data/`: `load_pairwise_examples`, `split_train_val`, `data/dataset.py`, `data/download.py`.
   - `models/lstm_baseline.py`
   - `models/sft_head.py`
   - `models/lora_head.py`
   - `train.py` + `evaluate.py` — does **not** wait on the three model sessions. Its tests build a
     fake `ModelBundle` by hand (a two-line `nn.Linear` model + a trivial `collate_fn` that returns
     random tensors of the right shape) to exercise the full checkpoint/resume state machine without
     any real HF model or tokenizer. This is the concrete payoff of Section B's contract: the
     train-loop session starts on day one, not after the model sessions finish.
3. Integration happens only at `models/__init__.py`'s three-line import list and at
   `configs/*.yaml` (one file per variant, no shared editing) — both trivial merges.

## Error handling summary (ties to existing negative-space rules)

| Failure | Category | Handling |
|---|---|---|
| Unknown `variant:` in YAML | Operating error | `load_config` raises a typed exception |
| Resume checkpoint's config != current config | Operating error | `ConfigMismatchError`, raised in `train.py` |
| `collate_fn` output missing `"labels"` | Programmer error | Assert in `train.py` before the training loop starts (fail fast, not mid-epoch) |
| `model(**inputs)` wrong output shape/non-finite | Programmer error | Assert in the shared plumbing test every variant runs (per `pytest-expert`'s shape/dtype contract tests) — not asserted at every training step in production, which would be redundant per-batch cost for a property that does not change at runtime |
| Duplicate `@register(variant)` | Programmer error | Assert inside `register`, at import time |

## Testing seam (already established conventions, applied here)

- Data tests build `PairwiseExample` lists from the shared tiny synthetic CSV fixture
  (`CLAUDE.md`'s existing convention) and assert `split_train_val`'s disjointness/coverage
  postconditions.
- Each model variant's tests build a *real* (tiny, randomly-initialized) instance of its own model
  — never a downloaded checkpoint — and run the shape/dtype/gradient/one-step plumbing tests from
  `pytest-expert`'s `ml-testing` reference. Real HF/LoRA weight loading is `@pytest.mark.integration`
  only.
- `train.py`/`evaluate.py` tests use a fake `ModelBundle` (Section D) and a handful of
  `PairwiseExample`s built in-test — no real dataset, no real model, no network. This suite proves
  the checkpoint/resume/W&B-resume state machine independent of every other component.

## E. Model persistence: Hugging Face Hub upload

Added 2026-09-07, as a bounded addendum — none of Sections A-D change. RunPod's `/workspace` is
only a 50GB persistent volume shared across all three variants (Section C decisions), with no
separate Network Volume (`CLAUDE.md`'s RunPod section); pushing a finished run to the HF Hub is the
actual durable long-term store, not pod-local disk.

**Trigger**: a separate, opt-in `scripts/push_to_hub.py`, never called from `train.py`. Every
experimental run gets checkpointed locally regardless of quality (Section C); only a run you've
reviewed and decided is worth keeping gets pushed. This keeps `train.py`'s loop fully local and
network-free except for the (already-existing) W&B logging calls.

```python
def push_to_hub(checkpoint_path: Path, repo_id: str, private: bool = True) -> str:
    """Loads a Checkpoint, reconstructs the model via build_model(checkpoint.config), loads its
    state dict, pushes the model (+ tokenizer, for HF-backed variants) and a model card. Returns
    the resulting commit URL."""
```

- **Reconstruction reuses the existing registry seam** — `build_model(checkpoint.config)` gives
  back the right architecture; no new hook is added to `models/registry.py` or `ModelBundle`.
- **Push dispatch is by model capability, not by variant name** — consistent with the "plain
  `nn.Module`, no variant branching" decision in Section B:
  - `isinstance(model, peft.PeftModel)` → its own `push_to_hub()` uploads the **adapter only**
    (a few MB-tens of MB, not the merged base-model-sized weights — `merge_and_unload()` is
    explicitly not used, since it would defeat LoRA's small-artifact point).
  - `isinstance(model, transformers.PreTrainedModel)` (the SFT+head variant) → same method,
    uploads full weights.
  - Otherwise (the LSTM baseline — a plain custom `nn.Module` with no Hub-native save format) →
    `huggingface_hub.upload_file` with a plain `torch.save`'d state dict.
- **Tokenizer**: for variants with `config.hf_model_name` (SFT+head, LoRA), also push
  `AutoTokenizer.from_pretrained(config.hf_model_name)` — derived fresh from the config already in
  the checkpoint, not stored anywhere new. The LSTM baseline has no HF tokenizer to push.
- **Model card**: a generated `README.md` embedding `checkpoint.config` (the exact hyperparameters
  that produced this weight file) and `checkpoint.best_val_metric` — the same reproducibility
  reasoning as storing `config` inside `Checkpoint` itself (Section C). This is also where the
  privacy decision below earns its keep: a model card's example section is exactly where profane or
  vulgar competition text could otherwise leak into a public artifact.
- **Repos**: one private repo per variant — `objones25/llm-reward-lstm-baseline`,
  `objones25/llm-reward-small-sft-head`, `objones25/llm-reward-medium-lora`. Each `push_to_hub`
  call is a new commit on that repo; HF's own commit history is the run history, so no separate
  versioning scheme is needed. Private by default (`private=True`), overridable via a CLI flag if a
  variant is ever deliberately made public later.

## Non-goals (explicit, so a future session doesn't relitigate them)

- Step-level / mid-epoch resume.
- A separate RunPod Network Volume (already decided in `CLAUDE.md`: template's `/workspace` only —
  the Hub upload in Section E is what replaces the need for one).
- A richer model `Protocol` beyond plain `nn.Module` (no `compute_loss`/`parameters_to_optimize`).
- Per-epoch checkpoint retention (only `last.pt` + `best.pt` exist).
- `submit.py`'s own prediction/inference contract — a follow-up design once training exists.
- Automatic/every-run Hub uploads, and merged (non-adapter) uploads for the LoRA variant.

## Open questions

None blocking. One follow-up worth flagging when `submit.py` is designed: whether it re-tokenizes
`test.csv` through the same `bundle.collate_fn` used in training (likely yes, for consistency) —
deferred per the stated non-goal above.
