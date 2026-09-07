# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Kaggle "LLM Classification Finetuning" competition: given a prompt and two LLM responses
(`response_a`, `response_b`), predict which one a human judge preferred. The label is **3-way**
(`winner_model_a` / `winner_model_b` / `winner_model_tie`), not strictly binary — model heads and
loss functions must account for the tie class, even though the exploratory framing was "reward
model." Score is a held-out test set (~25K rows); this is a Code Competition, so the submission
pipeline must run standalone against a replaced `test.csv`.

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
uv run pytest tests/unit/test_pairwise.py::test_tie_label_is_preserved   # single test
uv run pytest -m integration             # integration suite: hits Kaggle/HF/RunPod for real, opt-in only
uv run pytest -m gpu                     # GPU-marked tests, opt-in only

uv run ruff check .                      # lint
uv run python scripts/audit_negative_space.py src/ --select NSP002,NSP003,NSP005,NSP006,NSP007  # hard gate
uv run python scripts/download_data.py   # kagglehub competition_download, writes to data/raw/

uv run python -m llm_reward.train --config configs/lstm_baseline.yaml
uv run python -m llm_reward.train --config configs/small_sft_head.yaml
uv run python -m llm_reward.train --config configs/medium_lora.yaml
uv run python -m llm_reward.submit --config configs/<same-config> --checkpoint <path>  # writes submission.csv
```

`.env` (already present, loaded via `python-dotenv` at process entry only — never inside library
code) holds `KAGGLE_API_TOKEN`, `HF_TOKEN`, `WANDB_API_KEY`. `KAGGLE_API_TOKEN` is kagglehub's
modern single-token env var (highest priority in its credential discovery order); don't add
`KAGGLE_USERNAME`/`KAGGLE_KEY` unless the token approach stops working.

## Architecture

**Layout:** single `src/llm_reward/` package (src-layout), not per-experiment folders — the three
model variants share almost everything except model construction, so a shared package with one
variation point avoids duplicating the data/train/eval/submit path three times (DRY, YAGNI).

```
src/llm_reward/
  config.py        # dataclasses (one per variant) loaded from configs/*.yaml — YAML because RunPod
                    # jobs and future W&B sweeps need file-based configs, not just CLI args
  data/
    download.py     # thin wrapper around kagglehub.competition_download — the only place that touches
                    # the network for data; everything downstream takes a local path
    pairwise.py     # train.csv/test.csv -> (prompt, response_a, response_b, label) records;
                    # owns tie handling and any prompt/response truncation policy
    dataset.py      # torch Dataset/collate_fn over pairwise records, shared by all 3 variants
  models/
    registry.py     # name -> build_model(config) dispatch; the ONE place that knows about all
                    # three variants — train.py/submit.py never branch on model type themselves
    lstm_baseline.py
    sft_head.py     # small HF model + classification head
    lora_head.py    # medium HF model + peft LoRA + classification head
  train.py          # single entrypoint for all variants: load config -> registry.build_model -> fit
  evaluate.py
  submit.py         # writes submission.csv matching sample_submission.csv's schema exactly
configs/
  lstm_baseline.yaml
  small_sft_head.yaml
  medium_lora.yaml
scripts/
  download_data.py
  runpod_train.sh     # documents the expected pod environment (CUDA, uv) and SSH invocation
```

**Model registry is the DRY seam.** `train.py`, `evaluate.py`, and `submit.py` are variant-agnostic;
adding a fourth model idea means adding one `models/*.py` + one registry entry + one YAML config,
never touching the data pipeline or the other two variants.

**RunPod workflow is manual, not scripted.** You provision/start/stop the pod yourself (RunPod
CLI/MCP or console); this repo only needs to run cleanly over SSH once `uv sync` has been run on the
pod. `scripts/runpod_train.sh` documents that invocation — it is not a provisioning tool.

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
- Programmer errors (an impossible internal state — e.g. registry returns a model with the wrong
  output width) assert and crash. Operating errors (Kaggle download fails, HF Hub is unreachable,
  a checkpoint path doesn't exist) raise a typed exception and get handled at the call site — never
  conflate the two, and never swallow the operating-error case silently.
