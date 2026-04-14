#!/usr/bin/env bash
set -euo pipefail

# One-shot runner:
# - checkout + pull branch
# - auto-build RUN_ID from branch/gpu/iters
# - start telemetry
# - run training
# - stop telemetry
# - save artifacts + commit + push via save.sh
#
# Usage:
#   bash scripts/run_train_and_save.sh <branch> <iterations> [run_number]
#
# Example:
#   bash scripts/run_train_and_save.sh stf-minimal 2000 1

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/run_train_and_save.sh <branch> <iterations> [run_number]" >&2
  exit 1
fi

BRANCH="$1"
ITERATIONS="$2"
RUN_NUMBER="${3:-1}"

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "error: iterations must be an integer, got '$ITERATIONS'" >&2
  exit 1
fi
if ! [[ "$RUN_NUMBER" =~ ^[0-9]+$ ]]; then
  echo "error: run_number must be an integer, got '$RUN_NUMBER'" >&2
  exit 1
fi

git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
GPU_SLUG="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\+/-/g; s/^-//; s/-$//')"

export RUN_ID="$(date +%F)_${BRANCH}_${GPU_SLUG}_${GPU_COUNT}gpu_i${ITERATIONS}_run${RUN_NUMBER}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024/}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export ITERATIONS="$ITERATIONS"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"

echo "branch=$BRANCH"
echo "run_id=$RUN_ID"
echo "nproc_per_node=$NPROC_PER_NODE"
echo "iterations=$ITERATIONS"

chmod +x scripts/capture_telemetry.sh
RUN_ID="$RUN_ID" TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-15}" ./scripts/capture_telemetry.sh &
TELEMETRY_PID=$!

cleanup() {
  if kill -0 "$TELEMETRY_PID" >/dev/null 2>&1; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py

cleanup
trap - EXIT

bash save.sh

echo "done: $RUN_ID"
