# llm-reward

Predicting which of two LLM responses a human judge preferred, across three model
architectures of increasing cost.

## Background

This implements a solution to Kaggle's [LLM Classification Finetuning
competition](https://www.kaggle.com/competitions/llm-classification-finetuning): given a
prompt and two candidate responses (`response_a`, `response_b`) from two different LLMs, predict
which one a human judge preferred — or whether they called it a tie. The label is genuinely
3-way (`winner_model_a` / `winner_model_b` / `winner_model_tie`) — worth stating explicitly,
since the framing as "reward modeling" reads as a binary preference at a glance.

Rather than committing to one architecture, the project trains and compares three, in increasing
cost order:

1. **LSTM baseline** — trained from scratch, no pretrained weights. The cheap reference point
   every other variant has to beat.
2. **Small open-weight SFT model + classification head** (`Qwen/Qwen2.5-0.5B-Instruct`) — full
   fine-tune, leveraging a model already instruction-tuned before adding a 3-way head.
3. **Medium SFT model + LoRA** (`Qwen/Qwen2.5-7B-Instruct`) — parameter-efficient fine-tuning on a
   larger backbone, trained on a rented GPU pod rather than locally.

All three share one data pipeline, one training loop, and one evaluation path, and differ only in
how the model itself is built — see [Architecture](#architecture).

## Status

**Training has not been run yet.** The full pipeline — data loading, all three model variants,
the training/checkpointing/resume loop, and an opt-in Hugging Face Hub push — is implemented and
tested (92 tests, synthetic fixtures only, no network). What's *not* done: downloading the real
competition data, and training any of the three variants to get real metrics. There are
no accuracy numbers to report here yet — when there are, this section is where they'll go, per
variant, alongside the RunPod GPU-hours each one took.

## Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd llm-classification-tuning
uv sync
uv run pytest -q
```

```text
92 passed, 3 deselected in 5.28s
Total coverage: 94.08%
```

The 3 deselected tests need real network access (Kaggle/Hugging Face) and are opt-in by design —
see [Testing](#testing).

Create a `.env` file in the repo root (never committed — see `.gitignore`) with:

| Variable | Used for |
|---|---|
| `KAGGLE_API_TOKEN` | Downloading the competition data with `kagglehub` |
| `HF_TOKEN` | Loading the two Hugging Face model variants, and pushing trained checkpoints to the Hub |
| `WANDB_API_KEY` | Experiment tracking during training |

## Usage

Download the competition data (needs `KAGGLE_API_TOKEN`, and accepted competition rules on
Kaggle):

```python
import kagglehub

path = kagglehub.competition_download("llm-classification-finetuning")
```

Train one of the three variants:

```bash
uv run python -m llm_reward.train --config configs/lstm_baseline.yaml
uv run python -m llm_reward.train --config configs/small_sft_head.yaml
uv run python -m llm_reward.train --config configs/medium_lora.yaml
```

Each writes `last.pt` (for resuming: add `--resume`) and `best.pt` under the config's
`output_dir`, and logs per-step/per-epoch metrics to Weights & Biases. Once you've reviewed a
run's metrics and decided to keep it, push that checkpoint to a private Hugging Face Hub repo:

```bash
uv run python scripts/push_to_hub.py --checkpoint outputs/lstm_baseline/best.pt \
    --repo-id <your-username>/llm-reward-lstm-baseline
```

*(These three commands describe the real, tested code paths — the `train`/`push_to_hub`
invocations themselves haven't been run against the real dataset in this environment, since that
needs live Kaggle/Hugging Face credentials and, for the two larger variants, a GPU.)*

## Architecture

A model registry is the seam that lets the three variants share everything except model
construction: `train.py` and `evaluate.py` never branch on which variant they're running.

```text
src/llm_reward/
  data/          # kagglehub download -> PairwiseExample records -> train/val split -> Dataset
  models/
    config.py    # one config dataclass per variant, loaded from configs/*.yaml
    registry.py  # name -> build_model(config) dispatch -- the only place that knows all 3 variants
    lstm_baseline.py / sft_head.py / lora_head.py
  train.py       # single entrypoint for all variants: device placement, AMP, resume, W&B logging
  evaluate.py    # confusion-matrix-derived accuracy/precision/recall/F1
```

Full design rationale — why tokenization is per-variant, why resume excludes `epochs` from its
config-match check, the checkpoint/W&B contract, and more — lives in
[`CLAUDE.md`](CLAUDE.md) and the design docs under
[`docs/superpowers/specs/`](docs/superpowers/specs/).

The medium/LoRA variant targets a RunPod pod with an RTX PRO 6000 (96GB VRAM); see `CLAUDE.md`
for the exact template and disk layout.

## Testing

```bash
uv run pytest                 # full unit suite -- fast, no network, no GPU
uv run pytest --no-cov -m integration   # hits Kaggle/Hugging Face for real, opt-in only
uv run pytest --no-cov -m gpu           # CUDA-only tests
uv run ruff check .           # lint
```

Unit tests never touch the network or download real model weights: each model variant's tests
build a tiny, randomly-initialized instance of its own architecture (with `from_config`, not
`from_pretrained`) to test shapes, gradients, and one training step. See `tests/conftest.py`.

## Honest limits

- No trained checkpoints or accuracy numbers exist yet (see [Status](#status)).
- `submit.py` (writing `submission.csv` for the Kaggle leaderboard) isn't built — the training
  loop needed to exist first.
- The LSTM baseline pads every sequence to a fixed length and reads the final hidden state after
  the padding tail, rather than packing or masking. It's meant to be beaten, not optimal.
- Everything here is Kaggle-competition tooling for one specific dataset, not a general-purpose
  preference-model training library.

## Contributing

Personal Kaggle competition project — not set up to accept outside contributions.
Bug reports and questions are welcome as GitHub issues.

## License

No license has been chosen yet.
