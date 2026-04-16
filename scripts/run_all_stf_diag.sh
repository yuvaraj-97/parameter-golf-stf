#!/usr/bin/env bash
set -euo pipefail

# Run from /tmp so branch checkouts cannot replace/remove this script mid-run.
if [[ "${RUN_ALL_STF_FROM_TMP:-0}" != "1" ]]; then
  tmp_script="/tmp/run_all_stf_diag_$$.sh"
  cp "$0" "$tmp_script"
  chmod +x "$tmp_script"
  RUN_ALL_STF_FROM_TMP=1 exec bash "$tmp_script" "$@"
fi

REPO_DIR="${REPO_DIR:-/workspace/parameter-golf}"
ITERS="${1:-${ITERS:-500}}"
RUN_NUM="${2:-${RUN_NUM:-3}}"

cd "$REPO_DIR"

india_date() {
  TZ='Asia/Kolkata' date
}

finish_time() {
  echo
  echo "all_runs_finished_at_india_time:"
  india_date
}
trap finish_time EXIT

echo "all_runs_started_at_india_time:"
india_date
echo "repo_dir=${REPO_DIR}"
echo "iterations=${ITERS}"
echo "run_number=${RUN_NUM}"

python3 data/cached_challenge_fineweb.py --variant sp1024

TRAIN_SHARDS="$(ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l | tr -d ' ')"
echo "train_shards=${TRAIN_SHARDS}"

ensure_branch() {
  local branch="$1"

  if git show-ref --verify --quiet "refs/heads/${branch}"; then
    git checkout "$branch"
    git pull --ff-only origin "$branch"
  else
    git fetch origin "${branch}:${branch}"
    git checkout "$branch"
  fi
}

run_branch() {
  local branch="$1"
  local status=0

  echo
  echo "============================================================"
  echo "branch_started=${branch}"
  india_date
  echo "============================================================"

  set +e
  (
    set -euo pipefail
    ensure_branch "$branch"

    if [[ "$branch" == "stf-recurrence" ]]; then
      env \
        COMPILE_MODEL=0 \
        STF_TELEMETRY=1 \
        VAL_LOSS_EVERY=100 \
        TRAIN_LOG_EVERY=100 \
        TELEMETRY_INTERVAL_SECONDS=15 \
        TRAIN_BATCH_TOKENS=524288 \
        bash scripts/run_train_and_save.sh "$branch" "$ITERS" "$RUN_NUM"
    else
      env \
        COMPILE_MODEL=0 \
        STF_TELEMETRY=1 \
        VAL_LOSS_EVERY=100 \
        TRAIN_LOG_EVERY=100 \
        TELEMETRY_INTERVAL_SECONDS=15 \
        bash scripts/run_train_and_save.sh "$branch" "$ITERS" "$RUN_NUM"
    fi
  )
  status=$?
  set -e

  echo
  echo "============================================================"
  echo "branch_finished=${branch}"
  echo "branch_status=${status}"
  india_date
  echo "============================================================"

  return "$status"
}

run_branch baseline-repro
run_branch stf-learned-gate
run_branch stf-recurrence
run_branch stf-minimal
run_branch stf-soft-freeze
run_branch stf-reactivation
run_branch stf-quantization
run_branch stf-budget-regularization
