#!/usr/bin/env bash
# Train on the RunPod pod described in CLAUDE.md (runpod-torch-v280 template, RTX PRO 6000).
# You provision/start/stop the pod yourself -- this script only documents the invocation that
# should run cleanly over SSH once you're checked out under /workspace.
#
# Usage: scripts/runpod_train.sh <config.yaml> [extra train.py args...]
# Example: scripts/runpod_train.sh configs/lstm_baseline.yaml
#          scripts/runpod_train.sh configs/medium_lora.yaml --resume

set -euo pipefail

WORKSPACE_ROOT="/workspace"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Everything that must survive a pod stop/start -- the venv, HF model cache, kagglehub cache,
# and checkpoints -- has to live on the persistent volume at /workspace, not the 30GB ephemeral
# container disk (CLAUDE.md, "Architecture"). Fail fast rather than silently lose all of that on
# the next stop/start.
case "$REPO_ROOT" in
  "$WORKSPACE_ROOT"|"$WORKSPACE_ROOT"/*) ;;
  *)
    echo "Error: expected this repo to be checked out under $WORKSPACE_ROOT (got $REPO_ROOT)." >&2
    echo "See CLAUDE.md's disk layout note before running this on a RunPod pod." >&2
    exit 1
    ;;
esac

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra train.py args...]" >&2
  exit 1
fi

CONFIG="$1"
shift

# HF_HOME and KAGGLEHUB_CACHE (kagglehub's actual cache-dir env var, verified against the
# installed library, not guessed) both default under $HOME -- redirect them onto the
# persistent volume so a stop/start doesn't force a multi-GB re-download.
export HF_HOME="$WORKSPACE_ROOT/.cache/huggingface"
export KAGGLEHUB_CACHE="$WORKSPACE_ROOT/.cache/kagglehub"
mkdir -p "$HF_HOME" "$KAGGLEHUB_CACHE"

cd "$REPO_ROOT"
uv sync

if [[ ! -f data/raw/train.csv ]]; then
  uv run python scripts/download_data.py
fi

uv run python -m llm_reward.train --config "$CONFIG" "$@"
