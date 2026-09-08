# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle "LLM Classification Finetuning" competition: given a prompt and two LLM responses
(`response_a`, `response_b`), predict which one a human judge preferred. The label is **3-way**
(`winner_model_a` / `winner_model_b` / `winner_tie`) — note the asymmetry: the tie column in both
`train.csv` and `sample_submission.csv` is `winner_tie`, never `winner_model_tie`, despite the
other two columns carrying the `_model_` infix. Not strictly binary — model heads and loss
functions must account for the tie class, even though the exploratory framing was "reward model."
Score is a held-out test set (~25K rows); this is a Code Competition, so the submission pipeline
must run standalone against a replaced `test.csv`.

Three model variants are being compared, in increasing cost order:

1. **LSTM baseline** — trained from scratch, no pretrained weights.
2. **Small open-weight SFT model + classification head** — full fine-tune or head-only.
3. **Medium/medium-large SFT model + classification head, trained with LoRA**.

All three consume the same pairwise schema and produce the same 3-class output, so they must share
one dataset/eval/submission path and differ only in the model-building step (see Architecture).

## Commands

Package management is **uv**; there is no separate build step.

```bash
uv sync                                  # install/sync the environment from pyproject.toml + uv.lock
uv add <package>                         # add a runtime dependency
uv add --dev <package>                   # add a dev/test-only dependency

uv run pytest                            # full unit suite (fast, no network, no GPU)
# single test — `--no-cov` because the suite-wide `--cov-fail-under=90` gate in addopts would
# otherwise fail any partial run
uv run pytest --no-cov tests/data/test_pairwise_loading.py::test_load_pairwise_examples_parses_all_rows
uv run pytest --no-cov -m integration    # integration suite: hits Kaggle/HF/RunPod for real, opt-in only
uv run pytest --no-cov -m gpu            # GPU-marked tests, opt-in only

uv run ruff check .                      # lint

uv run python -m llm_reward.train --config configs/lstm_baseline.yaml
uv run python -m llm_reward.train --config configs/small_sft_head.yaml
uv run python -m llm_reward.train --config configs/medium_lora.yaml

# one-time (or re-run to refresh): download the competition CSVs into data/raw/
uv run python scripts/download_data.py

# generate submission.csv from a trained checkpoint -- reads the variant straight out of
# checkpoint.config, so only --checkpoint is needed (no separate --config to keep in sync)
uv run python -m llm_reward.submit --checkpoint outputs/lstm_baseline/best.pt

# opt-in, after reviewing a run's metrics: push that checkpoint to the HF Hub
uv run python scripts/push_to_hub.py --checkpoint outputs/lstm_baseline/best.pt \
    --repo-id objones25/llm-reward-lstm-baseline
```

`.env` (already present, loaded with `python-dotenv` at process entry only — never inside library
code) holds `KAGGLE_API_TOKEN`, `HF_TOKEN`, `WANDB_API_KEY`. `KAGGLE_API_TOKEN` is kagglehub's
modern single-token env var (highest priority in its credential discovery order); don't add
`KAGGLE_USERNAME`/`KAGGLE_KEY` unless the token approach stops working.

## Architecture

**Layout:** single `src/llm_reward/` package (src-layout), not per-experiment folders — the three
model variants share almost everything except model construction, so a shared package with one
variation point avoids duplicating the data/train/eval/submit path three times (DRY, YAGNI).

```text
src/llm_reward/
  negative_space.py # require()/bounded()/check_shape()/check_finite() — fail-fast helpers used
                    # everywhere instead of bare `assert`, which `python -O` strips silently
  data/
    download.py     # thin wrapper around kagglehub.competition_download — the only place that touches
                    # the network for data; everything downstream takes a local path
    pairwise.py     # train.csv -> PairwiseExample records; owns tie handling, the train/val split,
                    # and any prompt/response truncation policy
    dataset.py      # torch Dataset over pairwise records, shared by all 3 variants
  models/
    config.py       # TrainConfig + one frozen subclass per variant, loaded from configs/*.yaml
    registry.py     # name -> build_model(config) dispatch; the ONE place that knows about all
                    # three variants — train.py never branches on model type itself
    checkpoint.py   # Checkpoint dataclass + ConfigMismatchError, torch.save/load'd for resume
    _common.py      # format_input(), shared by all three variants so their training text can't
                    # silently diverge
    _hf_common.py   # LogitsOnly wrapper + collate_fn shared by the two transformers-backed
                    # variants only (sft_head.py, lora_head.py) — lstm_baseline.py doesn't use this
    lstm_baseline.py
    sft_head.py     # small HF model + classification head
    lora_head.py    # medium HF model + peft LoRA + classification head
  train.py          # single entrypoint for all variants: load config -> registry.build_model -> fit
  evaluate.py
  submit.py         # checkpoint -> registry.build_model -> run over test.csv -> submission.csv;
                    # variant comes from checkpoint.config, so no separate --config flag needed
configs/
  lstm_baseline.yaml
  small_sft_head.yaml
  medium_lora.yaml
scripts/
  download_data.py  # download_competition_data() -> copy train/test/sample_submission CSVs into
                    # data/raw/ (the local path train.py/submit.py default to), then by default
                    # append lmarena-ai/arena-human-preference-100k onto train.csv
                    # (--no-include-external to skip); see the External Data note below
  push_to_hub.py    # opt-in: push a reviewed checkpoint's weights (+ tokenizer, + model card) to
                    # the HF Hub — the durable store, since /workspace doesn't survive pod deletion
  runpod_train.sh   # RunPod-only entrypoint: fails fast unless checked out under /workspace,
                    # points HF_HOME/KAGGLEHUB_CACHE there, uv sync, download_data.py if needed,
                    # then `uv run python -m llm_reward.train --config <config.yaml> [args...]`
```

**Model registry is the DRY seam.** `train.py`, `evaluate.py`, and `submit.py` are all
variant-agnostic; adding a fourth model idea means adding one `models/*.py` + one registry entry
+ one YAML config, never touching the data pipeline or the other two variants.

**`submit.py`'s id-order contract.** `test.csv`'s row order matched `sample_submission.csv`'s in
the one archive probed so far, but `submit.py` asserts this rather than assuming it: it reorders
predictions to `sample_submission.csv`'s id order and raises if the id sets don't match exactly.
It also outputs class **probabilities** (softmax over the 3 logits), not hard labels — the real
`sample_submission.csv` columns are `id,winner_model_a,winner_model_b,winner_tie` (same tie-column
name `train.csv` uses — see the "Project" section above for the naming asymmetry).

**Test-time examples use a sentinel label.** `test.csv` has no ground truth, but
`PairwiseExample.label` is a mandatory int consumed by every collate_fn. `load_test_examples`
(in `data/pairwise.py`) sets `label=-1` on every row rather than introducing a second,
parallel "unlabeled" type + collate_fn per variant; `-1` is documented on the field and accepted
by `PairwiseExample.__post_init__`. `submit.py` never reads the `labels` tensor this produces —
it drops that key from the batch before the forward pass.

**Every CSV's header is validated before its rows are.** `data/pairwise.py`'s
`require_csv_header(csv_path, expected_columns)` is the one shared check: `load_pairwise_examples`
validates against `TRAIN_COLUMNS`, `load_test_examples` against `TEST_COLUMNS`, and `submit.py`
against `{"id", *SUBMISSION_COLUMNS}` — all three raise a clear `CheckFailed` naming the mismatch
instead of a bare `KeyError` from a `row[...]` access deep in a parsing loop (a real crash on the
first RunPod run, before this check existed, over the `winner_tie`/`winner_model_tie` asymmetry
noted above). `scripts/download_data.py` runs the same check on all three files after download,
before copying them into `data/raw/` — so a schema change on Kaggle's end is caught at download
time, not partway through a training run.

**External Data: `arena-human-preference-100k` is appended onto `train.csv` by default.**
Permitted under the competition's External Data rule (Section 5): its prompts are CC-BY-4.0 and
its model outputs are governed by each model provider's own terms -- the same licensing
arrangement the competition's own training data is already under, not a new risk category.
`scripts/download_data.py`'s `_convert_external_row` maps its schema onto ours: `question_id`→
`id`, `model_a`/`model_b` pass through unchanged, `conversation_a`/`conversation_b` (lists of
`{role, content}` turn dicts) get reduced to `prompt`/`response_a`/`response_b`, and `winner`
(verified with a real sample to take 4 distinct values: `model_a`, `model_b`, `tie`, and
`tie (bothbad)` — the last two are genuinely separate strings, not pre-merged) gets one-hot
encoded, `tie` and `tie (bothbad)` both mapping to `winner_tie`. The extracted turns are
JSON-encoded (`json.dumps(turns)`), matching train.csv's own on-disk encoding, so
`data/pairwise.py`'s `_extract_text` parses this data through the exact same `json.loads` path as
real Kaggle rows — emitting pre-joined plain text instead would make ordinary prose fall through
to `_extract_text`'s `ast.literal_eval` fallback and reintroduce `SyntaxWarning` spam on content
never meant to go through it. After appending, ids are checked for uniqueness across the combined
file (`require(len(ids) == len(set(ids)), ...)`). `download_to_data_raw` always re-copies a fresh
`train.csv` from the Kaggle cache before appending, so re-running `download_data.py` is still
idempotent — it never accumulates duplicate external rows.

**`csv.field_size_limit` is raised, not left at the stdlib default.** `data/pairwise.py` sets it
to 10,000,000 at import time (module-level, so every `csv.reader`/`DictReader` in the process
benefits, not only its own). The default (131,072 bytes) is too small for a real multi-turn LLM
conversation joined into one field — this crashed on the first real end-to-end run once
external data was added, and is exactly the kind of large-field problem the original Kaggle data
could eventually hit too. Raised, not removed (`sys.maxsize`), so a genuinely corrupt file still
fails loudly rather than reading unboundedly.

**The two HF-backed variants see chat-template-formatted input, not bare concatenated text.**
`_hf_common.py`'s `make_hf_collate_fn` wraps every example as
`[{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": format_input(ex)}]`
and tokenizes it through `tokenizer.apply_chat_template(..., add_generation_prompt=True)`, rather
than calling `tokenizer(format_input(ex))` directly on the raw string. This matters because
`Qwen/Qwen2.5-*-Instruct` are instruction/chat-tuned models: verified directly against the real
tokenizer that `<|im_start|>`/`<|im_end|>` are dedicated special tokens (ids 151644/151645,
outside the 151643-token base vocabulary) both models were SFT-trained to recognize as
role/turn boundaries, while a hand-picked delimiter like the old bare `[RESPONSE A]` tokenizes
into 5 meaningless subword fragments the model was never trained to treat specially. `_common.py`'s
`format_input` itself is untouched and still used as-is by `lstm_baseline.py`, which has no
chat-template concept — only `_hf_common.py` wraps its output through the chat template.
`SYSTEM_MESSAGE` overrides the model's generic default ("you are a helpful assistant") with a
task-specific instruction, since a chat-tuned model's own instruction-following behavior is the
mechanism that makes it treat "compare these two responses" as the actual task rather than an
unstructured blob to react to.

Using an instruction-tuned checkpoint as a classifier backbone at all (rather than the base,
non-instruct model) is deliberate, not an oversight: `AutoModelForSequenceClassification` only
reuses the transformer backbone's learned representations, replacing the LM head with a fresh
classification head — reward models in the RLHF literature (InstructGPT, Anthropic's HH-RLHF,
Llama 2) are routinely initialized from SFT checkpoints for exactly this reason, since a model
that already represents "what a good answer looks like" transfers well to judging one.

**Best-checkpoint criterion is val_loss, and early stopping is opt-in per config.** `best.pt` is
saved whenever `val_loss` improves (not `val_accuracy` — a real run's accuracy stayed flat/noisy
across all 5 epochs while loss moved cleanly, so loss is the metric that shows when a run has
started overfitting). `TrainConfig.early_stopping_patience` (default `None`, disabled)
stops the run once `val_loss` hasn't improved for that many consecutive epochs.
`Checkpoint.epochs_without_improvement` carries the patience counter across `--resume` so a
resumed run continues counting instead of resetting to 0. **Checkpoints saved before this
feature existed are incompatible for resume**: their `best_val_metric` holds an accuracy value,
which this code now reads as a loss — delete old `outputs/*/{last,best}.pt` rather than resume
from them.

**RunPod workflow is manual, not scripted.** You provision/start/stop the pod yourself (RunPod
CLI/MCP or console) and run `scripts/runpod_train.sh <config.yaml>` over SSH once `uv sync` can
reach the pod; the script itself refuses to do anything unless the checkout is under
`/workspace` (the disk layout the next paragraph describes).

**Target pod: `runpod-torch-v280` template (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
CUDA 12.8.1 / torch 2.8.0 / Ubuntu 24.04) on an RTX PRO 6000 (Blackwell, 96GB VRAM).** `torch` in
`pyproject.toml` is pinned to `==2.8.0` and routed through the `pytorch-cu128` index on Linux
(`[tool.uv.sources]`) to match this exact build — Blackwell needs torch>=2.8 for solid sm_120
kernel support, and matching the pod's baked-in version avoids a second ~5GB CUDA-library download
on top of what the image already ships.

The pod's 80GB disk is **not** one volume: 30GB ephemeral container disk (OS/CUDA/preinstalled
torch — gone on pod deletion) + 50GB persistent volume mounted at `/workspace` (survives stop/start
of that pod, but not pod deletion — no separate Network Volume is used for this project, by
choice). Everything that needs to survive a stop/start — the repo clone, the `uv` venv, the HF
model cache, the kagglehub download, and checkpoints — must live under `/workspace`, not the
container disk: point `HF_HOME` and the kagglehub cache dir there, and run `uv sync`/`uv run` from
a `/workspace`-rooted checkout.

**Model persistence is the Hugging Face Hub, opt-in, never automatic.** Since there's no separate
Network Volume, a run's local checkpoint is gone once its pod is deleted. `scripts/push_to_hub.py`
is a standalone script (never called from `train.py`) that, once you've reviewed a run's eval
metrics and decided to keep it, reconstructs the model through `registry.build_model(checkpoint.config)`
and pushes it to a private per-variant repo (`objones25/llm-reward-{variant}`): the LoRA variant
pushes its adapter only (never `merge_and_unload()`'d — that would defeat the point of a small
artifact), the SFT+head variant pushes full weights with `transformers`' native `push_to_hub`, and
the LSTM baseline (a plain `nn.Module` with no Hub-native save format) falls back to
`huggingface_hub.upload_file` with a raw `state_dict`. See
`docs/superpowers/specs/2026-09-07-training-seams-design.md` (Section E) for the full contract.

### Testing (pytest, per the pytest-expert skill — apply its rules directly, don't re-derive them)

- Root `tests/conftest.py`: seeds Python/NumPy/torch RNGs, blocks real network calls by default, and
  defines **one shared fixture** for a tiny synthetic CSV matching the `train.csv` schema (a handful
  of rows, including at least one tie). All three model variants' unit tests build their
  `dataset.py`/`pairwise.py` cases from this single fixture — do not let each variant grow its own
  copy; that's the DRY failure mode this project is explicitly avoiding.
- `kagglehub.competition_download`, HF model/tokenizer loading, and any RunPod call are only ever
  invoked through `data/download.py` and `models/registry.py`. Unit tests fake those two seams;
  nothing in the unit suite touches the network or downloads real weights.
- Default suite = plumbing + step tests only (shapes/dtypes/gradients, one optimizer step lowers
  loss) — milliseconds, run on every change. `@pytest.mark.slow` for overfit-one-batch checks,
  `@pytest.mark.integration` for real Kaggle/HF/RunPod calls, `@pytest.mark.gpu` for CUDA-only
  tests. All three are deselected by default; CI/manual runs opt in explicitly.
- Never assert on exact loss/accuracy values or real model quality in the unit suite — assert
  direction (`loss_after < loss_before`), finiteness, and shape/dtype/device contracts. Accuracy
  thresholds belong in `integration` tests against the real held-out split.

### Negative-space / fail-fast rules (apply throughout `src/llm_reward/`)

- Every function that touches data shape gets pre/postconditions: assert the pairwise dataset never
  loses or duplicates rows across the prompt/response_a/response_b/label join, assert train/val
  splits don't overlap, assert every label is one of the 3 known classes before it reaches a loss
  function.
- Bound everything with a numeric limit and assert it: max sequence length before truncation, max
  retries on a Kaggle/HF download, max training steps — no unbounded `while True`.
- Programmer errors (an impossible internal state — for example, registry returns a model with the wrong
  output width) assert and crash. Operating errors (Kaggle download fails, HF Hub is unreachable,
  a checkpoint path doesn't exist) raise a typed exception and get handled at the call site — never
  conflate the two, and never swallow the operating-error case silently.
