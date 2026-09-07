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

Six forks were resolved before writing this spec (the sixth added 2026-09-07); each is
load-bearing for what follows.

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
6. **`TrainConfig` holds only fields that change the result of training**, because every field on it
   is checked for equality on `--resume` (Section C). Anything purely operational — `num_workers`,
   device selection — must never go on `TrainConfig`: changing machines between runs would then
   wrongly block a legitimate resume. Operational knobs are plain CLI flags to `train.py`, never
   checkpointed, never compared. Added 2026-09-07, after finding it would otherwise be ambiguous
   which future fields belong where.

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

`models/config.py` — one frozen base, one frozen subclass per variant. **All `kw_only=True`** —
without it, `SFTHeadConfig`/`LoRAConfig` as drafted below actually raise `TypeError` at class
definition time (verified): `hf_model_name` has no default but would land, via dataclass field
flattening, after `TrainConfig`'s defaulted fields. `kw_only=True` removes field-ordering rules
entirely (every constructor argument becomes keyword-only), which is the standard fix for this
exact dataclass-inheritance pitfall, and does not change equality semantics (still needed for the
resume config-match check in Section C):

```python
@dataclass(frozen=True, kw_only=True)
class TrainConfig:
    variant: ClassVar[str]          # discriminator; overridden by each subclass
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
    class_weights: tuple[float, float, float] | None = None  # e.g. up-weight the tie class,
                                                              # typically a minority label in this data
    mixed_precision: Literal["no", "bf16"] = "no"  # bf16 needs no GradScaler (unlike fp16), so
        # fp16 isn't offered at all: the only hardware this project trains on for real (Blackwell,
        # via CLAUDE.md) wants bf16, and CPU/dev-machine runs are correctness tests, not throughput
        # runs, so "no" there costs nothing. Fewer states than the code could support, on purpose.

@dataclass(frozen=True, kw_only=True)
class LSTMConfig(TrainConfig):
    variant: ClassVar[str] = "lstm_baseline"
    max_seq_len: int = 256          # no pretrained model to match, so a plain default is fine here
    vocab_size: int = 30_000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2

@dataclass(frozen=True, kw_only=True)
class SFTHeadConfig(TrainConfig):
    variant: ClassVar[str] = "small_sft_head"
    hf_model_name: str
    max_seq_len: int                # required, no default: must be chosen to fit hf_model_name's
                                     # actual context window, not silently inherited from elsewhere
    freeze_backbone: bool = False
    head_lr: float | None = None    # None = one lr for backbone + head; set to opt into two groups
    gradient_checkpointing: bool = False

@dataclass(frozen=True, kw_only=True)
class LoRAConfig(TrainConfig):
    variant: ClassVar[str] = "medium_lora"
    hf_model_name: str
    max_seq_len: int                # required, same reasoning as SFTHeadConfig
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    head_lr: float | None = None    # None = one lr for LoRA adapter + head; set to opt into two groups
    gradient_checkpointing: bool = False

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
    param_groups: list[dict] | None = None  # optional torch optimizer param groups, e.g. a
        # lower lr for the backbone/adapter and config.head_lr for the head. None (the LSTM's
        # case, and the HF variants' case when head_lr is None) means train.py falls back to a
        # single flat group over every trainable parameter at config.lr.

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

**HF-backed variants must unwrap `.logits` before returning from `forward`.** Verified locally:
`transformers`/`peft` models return a `ModelOutput` object (e.g. `SequenceClassifierOutputWithPast`)
from `forward()`, not a bare tensor — calling `bundle.model(**inputs)` on a raw
`AutoModelForSequenceClassification`/`PeftModel` would silently break the "forward returns a
`[B, 3]` tensor" contract above. `sft_head.py` and `lora_head.py` must each wrap their model in a
small adapter before constructing `ModelBundle`:

```python
class _LogitsOnly(nn.Module):
    """Unwraps a HF ModelOutput so forward(**inputs) returns a bare logits tensor,
    matching the ModelBundle contract instead of transformers'/peft's wrapper object."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model  # exposes the real model for Section E's isinstance dispatch

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        return self.hf_model(**inputs).logits
```

Duplicated in both files rather than added to `registry.py` — same reasoning as `_format_input`
below: registry.py is meant to be stable after this design lands, and the wrapper is 6 lines.
Section E's capability dispatch checks `getattr(model, "hf_model", model)` first, so an
`isinstance` check still sees the real `PreTrainedModel`/`PeftModel` underneath the wrapper.

**`gradient_checkpointing`, when set, is `sft_head.py`'s/`lora_head.py`'s own responsibility**,
inside `build_model`, not `train.py`'s: `hf_model.gradient_checkpointing_enable()`, and —
whenever the backbone has any frozen parameter, which is always true for LoRA and true for
`SFTHeadConfig` when `freeze_backbone=True` — also `hf_model.enable_input_require_grads()`.
Without the latter, gradient checkpointing recomputes the frozen embedding layer's forward pass
with no grad-tracking input, and backward silently produces no gradient for anything downstream.
Confirmed against `peft`'s own docs: this is exactly what `prepare_model_for_kbit_training` does
for the quantized case, which this project doesn't use, so it's set directly rather than pulling
in that helper.

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
    global_step: int           # total training steps so far, across all epochs — carried across
                                # resume so the W&B step x-axis doesn't reset to 0 on a new process
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
ConfusionMatrix = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
# confusion_matrix[true_label][predicted_label] = count. The single source of truth every
# other classification metric below is derived from — never accumulated separately, so there
# is exactly one place a counting bug could hide, not five.

@dataclass(frozen=True)
class EvalMetrics:
    loss: float
    accuracy: float
    n_examples: int
    confusion_matrix: ConfusionMatrix
    per_class_precision: tuple[float, float, float]  # class 0 (a), 1 (b), 2 (tie)
    per_class_recall: tuple[float, float, float]
    per_class_f1: tuple[float, float, float]
    macro_f1: float  # unweighted mean of per_class_f1 — the one number that surfaces a model
        # that quietly gave up on the minority class (tie is typically rare in this data — see
        # class_weights in Section B), which aggregate accuracy alone hides. A class with zero
        # true or zero predicted examples reports 0.0 for the metrics that would otherwise
        # divide by zero — a plausible, non-erroneous state for a small eval split, not a bug.

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalMetrics: ...
```

`per_class_precision[c] = confusion_matrix[c][c] / sum(confusion_matrix[i][c] for i in range(3))`
(column sum — how often class `c` was predicted), `per_class_recall[c] = confusion_matrix[c][c] /
sum(confusion_matrix[c])` (row sum — how often class `c` was the true label), `per_class_f1[c]` the
harmonic mean of the two, each guarded against a zero denominator.

`train.py` entry point:

```
uv run python -m llm_reward.train --config configs/<variant>.yaml [--resume]
```

Loop shape:

1. `config = load_config(args.config)`; `bundle = build_model(config)`.
2. **Device, resolved once**: `device = torch.accelerator.current_accelerator(check_available=True)
   or torch.device("cpu")`, then `bundle.model.to(device)` (in place — `nn.Module.to()` never
   needs reassignment, unlike a tensor's). This one call works identically for the LSTM and both
   `_LogitsOnly`-wrapped HF models, since it's a plain `nn.Module` method — no variant branching.
   **This step did not exist in the first draft of this spec**: without it, every tensor a
   `collate_fn` produces stays wherever it was created (the CPU), and the RunPod GPU is never
   actually used — a correctness gap, not a performance one, caught by reviewing the plan against
   current PyTorch practice before implementation started.
3. Build train/val `DataLoader`s via `PairwiseDataset` + `bundle.collate_fn`.
4. Build the optimizer: `params = bundle.param_groups if bundle.param_groups is not None else
   filter(lambda p: p.requires_grad, bundle.model.parameters())`, then
   `optimizer = AdamW(params, lr=config.lr, weight_decay=config.weight_decay)` — PyTorch applies
   `lr=config.lr` as the default for any group in `param_groups` that doesn't set its own `"lr"`, so
   this one line is correct whether or not the bundle opted into differential learning rates.
   Build the scheduler from `config.lr_scheduler`/`config.warmup_ratio` and
   `len(train_loader) * config.epochs` — a function of config and dataset size only, needing no
   model-specific information, so this step is identical for all three variants.
5. Assemble the W&B run config once: `dataclasses.asdict(config) | {"variant": config.variant,
   "n_train_examples": len(train_examples), "n_val_examples": len(val_examples)}` — every
   hyperparameter plus the two numbers someone will otherwise have to ask about later.
6. If `--resume` and `{config.output_dir}/last.pt` exists:
   - Load the `Checkpoint` with `torch.load(last_path, weights_only=False, map_location=device)` —
     `map_location` matters here specifically: a checkpoint written on a CUDA pod, resumed later on
     a machine without CUDA (or a different GPU count), fails to deserialize without it.
   - `if dataclasses.replace(checkpoint.config, epochs=config.epochs) != config: raise
     ConfigMismatchError(...)` — `epochs` is deliberately excluded from the comparison. Resuming
     specifically to train for *more* epochs (edit the YAML's `epochs`, then `--resume`) is the
     standard workflow this whole mechanism exists for; comparing `epochs` too would make that
     workflow always raise. Every other field changing (`batch_size`, `lr`, ...) still must raise,
     since those invalidate the saved optimizer/scheduler state. (Caught during implementation: the
     naive `checkpoint.config != config` check was tested against exactly this scenario and failed.)
   - Restore `model_state`/`optimizer_state`/`scheduler_state`.
   - `start_epoch = checkpoint.epoch + 1`; `global_step = checkpoint.global_step`;
     `best_val_metric = checkpoint.best_val_metric`.
   - `run = wandb.init(id=checkpoint.wandb_run_id, resume="must", project="llm-reward", config=run_config)`.
7. Else: `start_epoch = 0`; `global_step = 0`; `best_val_metric = -inf`;
   `run = wandb.init(resume="allow", project="llm-reward", config=run_config)` with a freshly
   generated run id.
8. **Hold the `run` object returned by `wandb.init`; every later call is `run.log(...)`/`run.id`/
   `run.finish()`, never the bare `wandb.*` module-level functions.** The module-level form targets
   an implicit global run and breaks once a script does anything more than the simplest single-run
   case — irrelevant for tests too, since a test that monkeypatches the `wandb` **module** already
   controls what `wandb.init(...)` returns and can hand back a fake object with the same shape.
   Declare the two W&B x-axes right after `init` so per-step logging never needs to pass `step=`
   (a step out of order is silently dropped, no exception): `run.define_metric("train/global_step")`,
   `run.define_metric("train/*", step_metric="train/global_step")`, `run.define_metric("epoch")`,
   `run.define_metric("val/*", step_metric="epoch")`.
9. For `epoch in range(start_epoch, config.epochs)`: `bundle.model.train()`, then for each batch:
   - `batch = {k: v.to(device) for k, v in batch.items()}` — every tensor the `collate_fn` produced,
     moved by copy (never in place for a tensor) to match where the model now lives.
   - `inputs = {k: v for k, v in batch.items() if k != "labels"}`; `labels = batch["labels"]`.
   - `with torch.autocast(device.type, dtype=torch.bfloat16, enabled=config.mixed_precision == "bf16"):
     logits = bundle.model(**inputs); loss = F.cross_entropy(logits, labels, weight=class_weights)`
     — forward and loss only; `backward()` stays outside the `autocast` block (it reuses whatever
     dtype the matching forward op picked).
   - `optimizer.zero_grad()` -> `loss.backward()` -> clip -> `optimizer.step()` -> `scheduler.step()`.
     **Gradient-norm logging, and clipping, are two different questions** — clipping always stays a
     single combined operation over every trainable parameter (`grad_norm =
     nn.utils.clip_grad_norm_(params, config.max_grad_norm)`, unchanged from the earlier draft); but
     when `bundle.param_groups is not None` — always exactly two groups by convention,
     `param_groups[0]` the backbone/adapter and `param_groups[1]` the head, per Section B's
     `sft_head.py`/`lora_head.py` — also compute (never clip on) each group's own norm purely for
     observability: `_group_grad_norm(group["params"]) = torch.norm(torch.stack([p.grad.detach().norm()
     for p in group["params"] if p.grad is not None])).item()` (0.0 if the group has no grad at all).
     A single combined clip is the standard, safe default; a randomly-initialized head and a
     pretrained-or-LoRA backbone otherwise want very different learning rates (already handled via
     `head_lr`) and can have very different gradient scales worth *seeing* separately, which a single
     combined norm hides.
   - Running epoch accumulators, from the same forward pass already computed — no extra cost:
     `epoch_loss_sum += loss.item() * labels.shape[0]`; `epoch_correct += (logits.argmax(-1) ==
     labels).sum().item()`; `epoch_n += labels.shape[0]`.
   - `run.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0],
     "train/global_step": global_step})`, plus either `{"train/grad_norm": float(grad_norm)}` (when
     `bundle.param_groups is None`) or `{"train/grad_norm_backbone": ..., "train/grad_norm_head":
     ...}` (when it isn't) merged into the same `run.log` call; `global_step += 1`.
   - After the epoch: `metrics = evaluate(bundle.model, val_loader, device)`, then one `run.log` call
     carrying `epoch`, `train/epoch_loss` (`epoch_loss_sum / epoch_n`) and `train/epoch_accuracy`
     (`epoch_correct / epoch_n`) alongside `val/loss`, `val/accuracy`, `val/macro_f1`, and
     `val/precision_{a,b,tie}` / `val/recall_{a,b,tie}` / `val/f1_{a,b,tie}` from `metrics`' three
     per-class tuples — training and validation land on the same `epoch` x-axis specifically so the
     gap between them (the overfitting signal) is one glance on the same chart, not two lookups.
     `val/confusion_matrix` logs separately, as a `wandb.Table` built from `metrics.confusion_matrix`
     (3 rows, one per true class) — a table, not a scalar, so per-epoch only, never per-step. Also,
     only when on an accelerator: `torch.cuda.max_memory_allocated(device)` when `device.type ==
     "cuda"`, `torch.mps.current_allocated_memory()` when `device.type == "mps"` (`torch.accelerator`
     itself has no memory-accounting API in this project's pinned torch==2.8.0 — confirmed empty,
     that API was added in a later release — so the device-specific module is used directly), logged
     as `system/gpu_mem_allocated_mb`, cheap, and the number that answers "did this variant actually
     fit, and with how much room" on the RunPod pod.
   - `improved = metrics.accuracy > best_val_metric`; `best_val_metric = max(best_val_metric,
     metrics.accuracy)`.
   - Always write `Checkpoint(epoch=epoch, global_step=global_step, ..., wandb_run_id=run.id)` to
     `last.pt`. If `improved`, also write it to `best.pt`.
10. `run.finish()` after the loop (or wrap steps 8-9 in `with wandb.init(...) as run:` so a crash
    marks the run failed automatically instead of leaving it "running" forever on the dashboard —
    either is acceptable; the context-manager form is the one that survives an exception un-nudged).

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
  the checkpoint/resume/W&B-resume state machine independent of every other component, including
  both the `bundle.param_groups is None` fallback and a fake bundle that sets it. The test double
  standing in for the `wandb` module must return an object from `.init(...)` that supports the
  same shape `train.py` actually calls: `.log(...)`, `.id`, `.define_metric(...)`, and (if `train.py`
  uses the context-manager form) `__enter__`/`__exit__` — a fake that only supports module-level
  `wandb.log(...)` would pass tests that no longer match the code once `run.log(...)` replaces it.
  These tests run on CPU only (no GPU in CI), so `device` resolves to `torch.device("cpu")` and
  `torch.autocast("cpu", dtype=torch.bfloat16, enabled=...)` is exercised for real — bf16 autocast
  works on CPU, just without the throughput win, which is exactly what a correctness test needs.
- Constructing one instance of every `TrainConfig` subclass is itself a test, not just a type-check
  in passing: the `kw_only` fix in Section B exists because this exact hierarchy raised `TypeError`
  at class-definition time before the fix, which is a programmer error a test catches immediately
  rather than one discovered later when someone finally imports `models.config`.
- `evaluate()`'s precision/recall/F1 are all derived from one confusion matrix (never accumulated
  separately), so the one test worth writing deliberately is a confusion matrix with an empty row
  or column — a class with zero true examples, or zero predicted examples — asserting the affected
  metric reports `0.0` rather than raising `ZeroDivisionError`.

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
  Loads with `torch.load(checkpoint_path, weights_only=False, map_location="cpu")` — this script
  never needs a GPU at all (it only reconstructs weights to hand to the Hub), and `map_location`
  is what makes that true regardless of which device the checkpoint was originally saved from.
- **Push dispatch is by model capability, not by variant name** — consistent with the "plain
  `nn.Module`, no variant branching" decision in Section B. First unwrap:
  `real_model = getattr(bundle.model, "hf_model", bundle.model)` — undoes the `_LogitsOnly`
  wrapper from Section B for the two HF-backed variants; the LSTM's model has no `hf_model`
  attribute, so `getattr` falls back to itself unchanged. Then:
  - `isinstance(real_model, peft.PeftModel)` → its own `push_to_hub()` uploads the **adapter
    only** (a few MB-tens of MB, not the merged base-model-sized weights — `merge_and_unload()`
    is explicitly not used, since it would defeat LoRA's small-artifact point).
  - `isinstance(real_model, transformers.PreTrainedModel)` (the SFT+head variant) → same method,
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
- fp16/`GradScaler` support — bf16-only, per the `mixed_precision` field's own reasoning in Section B.
- `torch.compile` — not adopted yet. Revisit once the plain-eager loop is proven correct;
  compiling a `peft`-wrapped model has known rough edges not worth taking on alongside everything
  else in this plan.
- Multi-GPU / distributed training (`DistributedDataParallel`/FSDP2) — single-GPU RunPod pod, per
  `CLAUDE.md`.

## Open questions

None blocking. One follow-up worth flagging when `submit.py` is designed: whether it re-tokenizes
`test.csv` through the same `bundle.collate_fn` used in training (likely yes, for consistency) —
deferred per the stated non-goal above.
