# llm-reward

Predicting which of two LLM responses a human judge preferred, across three model
architectures of increasing cost.

## Background

This implements a solution to Kaggle's [LLM Classification Finetuning
competition](https://www.kaggle.com/competitions/llm-classification-finetuning): given a
prompt and two candidate responses (`response_a`, `response_b`) from two different LLMs, predict
which one a human judge preferred — or whether they called it a tie. The label is genuinely
3-way (`winner_model_a` / `winner_model_b` / `winner_tie`) — worth stating explicitly, since the
framing as "reward modeling" reads as a binary preference at a glance.

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

**All three variants have a completed training run**, and their checkpoints are pushed to private
Hugging Face Hub repos (`objones25/llm-reward-{lstm-baseline,small-sft-head,medium-lora}`). Final
validation accuracy, on a ~16.4K-row held-out split (3-way random baseline: 33%):

| Variant | Epochs | Val accuracy |
|---|---|---|
| LSTM baseline | 5 | ~38.6% |
| Small SFT + head (Qwen2.5-0.5B) | 3 | ~47.9% |
| Medium LoRA (Qwen2.5-7B) | 3 | ~53.0% |

Accuracy rises with cost, as expected. These are point estimates read off the plotted curves, not
exact logged values — regenerate the full per-epoch curves and confusion matrices from the source
W&B run with `scripts/plot_run_metrics.py <run_id> --name <variant>` (output goes to
`graphs/plots/<variant>/`, gitignored). The LSTM number in particular is worth reading next to the
class-collapse caveat in [Honest limits](#honest-limits) before trusting it at face value.

What's not done yet: generating and submitting `submission.csv` against the real ~25K-row Kaggle
test set (`submit.py` has only been run against small local samples so far, and only for the two
HF-backed variants) and Kaggle leaderboard scoring.

## Install

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this-repo>
cd llm-classification-tuning
uv sync
uv run pytest -q
```

```text
153 passed, 4 deselected in 11.22s
Total coverage: 95.38%
```

The 4 deselected tests need real network access (Kaggle/Hugging Face) and are opt-in by design —
see [Testing](#testing).

Create a `.env` file in the repo root (never committed — see `.gitignore`) with:

| Variable | Used for |
|---|---|
| `KAGGLE_API_TOKEN` | Downloading the competition data with `kagglehub` |
| `HF_TOKEN` | Loading the two Hugging Face model variants, and pushing trained checkpoints to the Hub |
| `WANDB_API_KEY` | Experiment tracking during training |

## Usage

Download the competition data into `data/raw/` (needs `KAGGLE_API_TOKEN` and accepted competition
rules on Kaggle). By default this also appends the CC-BY-4.0
[`lmarena-ai/arena-human-preference-100k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k)
dataset onto `train.csv`, permitted under the competition's External Data rule — pass
`--no-include-external` to skip that:

```bash
uv run python scripts/download_data.py
```

Train one of the three variants:

```bash
uv run python -m llm_reward.train --config configs/lstm_baseline.yaml
uv run python -m llm_reward.train --config configs/small_sft_head.yaml
uv run python -m llm_reward.train --config configs/medium_lora.yaml
```

Each writes `last.pt` (for resuming: add `--resume`) and `best.pt` under the config's
`output_dir`, and logs per-step/per-epoch metrics to Weights & Biases — see
`scripts/plot_run_metrics.py <run_id> --name <variant>` to turn a run into
loss/accuracy/confusion-matrix plots. Generate a Kaggle submission from a trained checkpoint:

```bash
uv run python -m llm_reward.submit --checkpoint outputs/lstm_baseline/best.pt
```

Once you've reviewed a run's metrics and decided to keep it, push that checkpoint to a private
Hugging Face Hub repo:

```bash
uv run python scripts/push_to_hub.py --checkpoint outputs/lstm_baseline/best.pt \
    --repo-id <your-username>/llm-reward-lstm-baseline
```

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
  submit.py      # checkpoint -> registry.build_model -> run over test.csv -> submission.csv
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

- None of the three variants has been scored on the real Kaggle leaderboard yet (see
  [Status](#status)) — the accuracy numbers above are against the local validation split only.
- The LSTM baseline's best-by-`val_loss` checkpoint is from epoch 0 of 5, and at that epoch it
  never once predicts `winner_model_b` — it's only distinguishing "a" from "tie". It does learn to
  predict all three classes by epoch 3–4, but `val_loss` (not accuracy) is the checkpoint-selection
  criterion (see `CLAUDE.md`), and on this run it picked the epoch that hadn't gotten there yet.
- 53% (the best variant, medium LoRA) is well above the 33% random baseline but not close to
  strong performance on this task — none of the three variants has had serious hyperparameter
  tuning yet.
- The LSTM baseline pads every sequence to a fixed length and reads the final hidden state after
  the padding tail, rather than packing or masking. It's meant to be beaten, not optimal.
- Everything here is Kaggle-competition tooling for one specific dataset, not a general-purpose
  preference-model training library.

## Contributing

Personal Kaggle competition project — not set up to accept outside contributions.
Bug reports and questions are welcome as GitHub issues.

## License

No license has been chosen yet.
