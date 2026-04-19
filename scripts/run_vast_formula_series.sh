#!/usr/bin/env bash
set -Eeuo pipefail

# Run STF formula variants serially on a persistent Vast.ai repo.
#
# Expected .env:
#   TELEGRAM_BOT_TOKEN=...
#   TELEGRAM_USER_ID=...
#
# Alert-only smoke test before STF_SCORE_FN is implemented:
#   STF_BRANCHES="stf-learned-gate" ITERATIONS=1 bash scripts/run_vast_formula_series.sh

ROOT_DIR="$(git rev-parse --show-toplevel)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env"
  set +a
fi

STF_BRANCHES="${STF_BRANCHES:-stf-learned-gate}"
STF_SCORE_FNS="${STF_SCORE_FNS:-l2 relative_l2 cosine direction}"
ITERATIONS="${ITERATIONS:-500}"
RUN_NUMBER="${RUN_NUMBER:-1}"
DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
VOCAB_SIZE="${VOCAB_SIZE:-1024}"
VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-50}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
STF_TELEMETRY="${STF_TELEMETRY:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MUON_BACKEND_STEPS="${MUON_BACKEND_STEPS:-3}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-0}"
TELEMETRY_INTERVAL_SECONDS="${TELEMETRY_INTERVAL_SECONDS:-15}"
FETCH_BEFORE_RUN="${FETCH_BEFORE_RUN:-1}"
REQUIRE_SCORE_FN_SUPPORT="${REQUIRE_SCORE_FN_SUPPORT:-1}"
AUTO_COMMIT_RESULTS="${AUTO_COMMIT_RESULTS:-0}"

CURRENT_BRANCH=""
CURRENT_SCORE_FN=""
CURRENT_RUN_ID=""
TELEMETRY_PID=""
GPU_COUNT="1"
GPU_NAME="unknown-gpu"
GPU_SLUG="unknown-gpu"
ALERT_SENT="0"
CURRENT_CONSOLE_LOG=""
read -r -a BRANCH_LIST <<< "$STF_BRANCHES"
read -r -a FORMULA_LIST <<< "$STF_SCORE_FNS"
BRANCH_TOTAL="${#BRANCH_LIST[@]}"
FORMULA_TOTAL="${#FORMULA_LIST[@]}"
VARIANT_TOTAL="$((BRANCH_TOTAL * FORMULA_TOTAL))"
CURRENT_BRANCH_INDEX="0"
CURRENT_FORMULA_INDEX="0"
CURRENT_VARIANT_INDEX="0"
SUCCESS_COUNT="0"
FAILED_COUNT="0"
FAILED_VARIANTS=""

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

notify() {
  local text="$1"
  printf '[%s] %s\n' "$(timestamp)" "$text"
  if [ -z "${TELEGRAM_BOT_TOKEN:-}" ] || [ -z "${TELEGRAM_USER_ID:-}" ]; then
    return 0
  fi
  curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_USER_ID}" \
    --data-urlencode "text=${text}" >/dev/null || true
}

progress_bar() {
  local current="$1"
  local total="$2"
  local width="${3:-10}"
  local filled
  local empty
  local out=""
  local i

  if [ "$total" -le 0 ]; then
    total=1
  fi
  filled=$((current * width / total))
  if [ "$current" -gt 0 ] && [ "$filled" -eq 0 ]; then
    filled=1
  fi
  if [ "$filled" -gt "$width" ]; then
    filled="$width"
  fi
  empty=$((width - filled))

  for i in $(seq 1 "$filled"); do out="${out}🟩"; done
  for i in $(seq 1 "$empty"); do out="${out}⬜"; done
  printf '%s' "$out"
}

run_context() {
  cat <<EOF
Progress: $(progress_bar "$CURRENT_VARIANT_INDEX" "$VARIANT_TOTAL") ${CURRENT_VARIANT_INDEX}/${VARIANT_TOTAL}
Branch: ${CURRENT_BRANCH:-not set} (${CURRENT_BRANCH_INDEX}/${BRANCH_TOTAL})
Formula: ${CURRENT_SCORE_FN:-not set} (${CURRENT_FORMULA_INDEX}/${FORMULA_TOTAL})
Run: ${CURRENT_RUN_ID:-not assigned}
GPU: ${GPU_COUNT}x ${GPU_NAME}
Iterations: ${ITERATIONS}
Pod: $(hostname)
EOF
}

append_failed_variant() {
  local item="$1"
  if [ -z "$FAILED_VARIANTS" ]; then
    FAILED_VARIANTS="$item"
  else
    FAILED_VARIANTS="${FAILED_VARIANTS}, ${item}"
  fi
}

failure_summary() {
  if [ -z "${CURRENT_CONSOLE_LOG:-}" ] || [ ! -f "$CURRENT_CONSOLE_LOG" ]; then
    return 0
  fi

  grep -E "RuntimeError:|OSError:|ValueError:|FileNotFoundError:|ChildFailedError|train_gpt.py FAILED|Root Cause|error:" "$CURRENT_CONSOLE_LOG" \
    | tail -n 8 \
    | sed 's/^[[:space:]]*//; s/\[rank[0-9]\]://g' \
    || true
}

notify_repeat() {
  local text="$1"
  local count="${2:-3}"
  local delay="${3:-5}"
  local i
  for i in $(seq 1 "$count"); do
    notify "${text} (${i}/${count})"
    if [ "$i" != "$count" ]; then
      sleep "$delay"
    fi
  done
}

fail_with_alerts() {
  local exit_code="${1:-1}"
  local message="$2"
  local summary
  cleanup_telemetry
  if [ "$ALERT_SENT" = "1" ]; then
    exit "$exit_code"
  fi
  ALERT_SENT="1"
  summary="$(failure_summary)"
  notify_repeat "STOPPED

Reason: ${message}
Exit: ${exit_code}

$(run_context)

Key error:
${summary:-No console summary available.}

Action: attend to the pod."
  exit "$exit_code"
}

cleanup_telemetry() {
  if [ -n "${TELEMETRY_PID:-}" ] && kill -0 "$TELEMETRY_PID" >/dev/null 2>&1; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  TELEMETRY_PID=""
}

on_error() {
  local exit_code="$?"
  fail_with_alerts "$exit_code" "Vast STF formula series crashed"
}
trap on_error ERR
trap cleanup_telemetry EXIT

require_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    fail_with_alerts 1 "${name} must be an integer, got '${value}'"
  fi
}

preflight_training_inputs() {
  local shard_count

  if [ ! -f "$TOKENIZER_PATH" ]; then
    fail_with_alerts 1 "missing tokenizer at ${TOKENIZER_PATH}"
  fi
  if [ ! -d "$DATA_PATH" ]; then
    fail_with_alerts 1 "missing dataset directory at ${DATA_PATH}"
  fi

  shard_count="$(find "$DATA_PATH" -maxdepth 1 -type f -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"
  if [ "$shard_count" = "0" ]; then
    fail_with_alerts 1 "no fineweb_train_*.bin shards found in ${DATA_PATH}"
  fi
}

gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
  fi
  GPU_SLUG="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\+/-/g; s/^-//; s/-$//')"
  NPROC_PER_NODE="${NPROC_PER_NODE:-$GPU_COUNT}"
}

archive_run() {
  local branch="$1"
  local score_fn="$2"
  local run_id="$3"
  local console_log="$4"
  local date_tag
  local head_sha
  local final_line
  local run_dir

  date_tag="$(date +%F)"
  head_sha="$(git rev-parse --short HEAD)"
  run_dir="vast/experiments/${date_tag}-${GPU_SLUG}-${GPU_COUNT}gpu/${branch}/${run_id}"
  mkdir -p "$run_dir"

  [ -f "logs/${run_id}.txt" ] && cp "logs/${run_id}.txt" "$run_dir/train.log"
  [ -f "$console_log" ] && cp "$console_log" "$run_dir/console.log"
  [ -f "logs/telemetry/${run_id}.csv" ] && cp "logs/telemetry/${run_id}.csv" "$run_dir/telemetry.csv"
  cp "train_gpt.py" "$run_dir/train_gpt.py"
  [ -f "final_model.pt" ] && cp "final_model.pt" "$run_dir/final_model.pt"
  [ -f "final_model.int8.ptz" ] && cp "final_model.int8.ptz" "$run_dir/final_model.int8.ptz"

  final_line="$(grep -E "final_int8_zlib_roundtrip_exact|final_int8_zlib_roundtrip|step:[0-9]+/[0-9]+ val_loss" "logs/${run_id}.txt" 2>/dev/null | tail -n 1 || true)"

  cat > "$run_dir/README.md" <<EOF
# ${run_id}

- Branch: ${branch}
- Commit: ${head_sha}
- STF score function: ${score_fn}
- Pod: $(hostname)
- GPU: ${GPU_COUNT}x ${GPU_NAME}
- Iterations: ${ITERATIONS}
- Run number: ${RUN_NUMBER}

## Final Metric Line

\`\`\`
${final_line}
\`\`\`
EOF

  if [ "$AUTO_COMMIT_RESULTS" = "1" ]; then
    git add "$run_dir"
    git commit -m "logs: add ${run_id} (${branch}, ${score_fn})"
  fi

  echo "$run_dir"
}

require_integer "ITERATIONS" "$ITERATIONS"
require_integer "RUN_NUMBER" "$RUN_NUMBER"
gpu_info
preflight_training_inputs

mkdir -p logs logs/telemetry vast/experiments

notify "SERIES STARTED

Branches (${BRANCH_TOTAL}): ${STF_BRANCHES}
Formulas (${FORMULA_TOTAL}): ${STF_SCORE_FNS}
Total variants: ${VARIANT_TOTAL}
Progress: $(progress_bar 0 "$VARIANT_TOTAL") 0/${VARIANT_TOTAL}
Iterations: ${ITERATIONS}
GPU: ${GPU_COUNT}x ${GPU_NAME}
Pod: $(hostname)"

for branch in "${BRANCH_LIST[@]}"; do
  CURRENT_BRANCH_INDEX="$((CURRENT_BRANCH_INDEX + 1))"
  CURRENT_BRANCH="$branch"
  if [ "$FETCH_BEFORE_RUN" = "1" ]; then
    git fetch origin "$branch"
  fi
  git checkout "$branch"
  if [ "$FETCH_BEFORE_RUN" = "1" ]; then
    git merge --ff-only "origin/${branch}"
  fi

  if [ "$REQUIRE_SCORE_FN_SUPPORT" = "1" ] && ! grep -q "stf_score_fn" train_gpt.py; then
    fail_with_alerts 1 "branch ${branch} does not appear to support STF_SCORE_FN yet"
  fi

  CURRENT_FORMULA_INDEX="0"
  for score_fn in "${FORMULA_LIST[@]}"; do
    variant_exit=0
    CURRENT_FORMULA_INDEX="$((CURRENT_FORMULA_INDEX + 1))"
    CURRENT_VARIANT_INDEX="$((CURRENT_VARIANT_INDEX + 1))"
    CURRENT_SCORE_FN="$score_fn"
    safe_branch="$(printf '%s' "$branch" | tr '/' '-')"
    CURRENT_RUN_ID="$(date +%F)_${safe_branch}_${score_fn}_${GPU_SLUG}_${GPU_COUNT}gpu_i${ITERATIONS}_run${RUN_NUMBER}"
    console_log="logs/${CURRENT_RUN_ID}.console.txt"
    CURRENT_CONSOLE_LOG="$console_log"

    rm -f "logs/${CURRENT_RUN_ID}.txt" "$console_log" "logs/telemetry/${CURRENT_RUN_ID}.csv" final_model.pt final_model.int8.ptz

    notify "VARIANT STARTED

$(run_context)"

    if [ -x "scripts/capture_telemetry.sh" ]; then
      RUN_ID="$CURRENT_RUN_ID" TELEMETRY_INTERVAL_SECONDS="$TELEMETRY_INTERVAL_SECONDS" ./scripts/capture_telemetry.sh &
      TELEMETRY_PID="$!"
    fi

    export RUN_ID="$CURRENT_RUN_ID"
    export DATA_PATH TOKENIZER_PATH VOCAB_SIZE
    export ITERATIONS VAL_LOSS_EVERY TRAIN_LOG_EVERY
    export STF_TELEMETRY WARMUP_STEPS MUON_BACKEND_STEPS MAX_WALLCLOCK_SECONDS
    export STF_SCORE_FN="$score_fn"

    if {
      echo "timestamp=$(timestamp)"
      echo "branch=${branch}"
      echo "run_id=${CURRENT_RUN_ID}"
      echo "score_fn=${score_fn}"
      echo "nproc_per_node=${NPROC_PER_NODE}"
      torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py
    } 2>&1 | tee "$console_log"; then
      variant_exit=0
    else
      variant_exit="$?"
    fi

    cleanup_telemetry

    run_dir="$(archive_run "$branch" "$score_fn" "$CURRENT_RUN_ID" "$console_log")"
    final_line="$(grep -E "final_int8_zlib_roundtrip_exact|final_int8_zlib_roundtrip|step:[0-9]+/[0-9]+ val_loss" "logs/${CURRENT_RUN_ID}.txt" 2>/dev/null | tail -n 1 || true)"
    if [ "$variant_exit" -ne 0 ]; then
      FAILED_COUNT="$((FAILED_COUNT + 1))"
      append_failed_variant "${branch}/${score_fn}"
      notify "VARIANT FAILED - CONTINUING

$(run_context)

Exit: ${variant_exit}
Archive: ${run_dir}

Key error:
$(failure_summary)

Next variant will start shortly."
      continue
    fi

    SUCCESS_COUNT="$((SUCCESS_COUNT + 1))"
    notify "VARIANT ENDED

$(run_context)

Archive: ${run_dir}
Final: ${final_line:-not found}"
    notify "MOVING TO NEXT

Completed: ${branch} / ${score_fn}
Progress: $(progress_bar "$CURRENT_VARIANT_INDEX" "$VARIANT_TOTAL") ${CURRENT_VARIANT_INDEX}/${VARIANT_TOTAL}
Next variant will start shortly."
  done
done

notify "SERIES DONE

Branches (${BRANCH_TOTAL}): ${STF_BRANCHES}
Formulas (${FORMULA_TOTAL}): ${STF_SCORE_FNS}
Progress: $(progress_bar "$VARIANT_TOTAL" "$VARIANT_TOTAL") ${VARIANT_TOTAL}/${VARIANT_TOTAL}
Succeeded: ${SUCCESS_COUNT}
Failed: ${FAILED_COUNT}
Failed variants: ${FAILED_VARIANTS:-none}
Iterations: ${ITERATIONS}
GPU: ${GPU_COUNT}x ${GPU_NAME}"
