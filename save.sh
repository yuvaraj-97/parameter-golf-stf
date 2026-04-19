#!/usr/bin/env bash
set -euo pipefail

# Save one completed run into vast/experiments/... and push it.
# Expected training outputs from this repo:
# - logs/${RUN_ID}.txt (required)
# - final_model.pt (optional)
# - final_model.int8.ptz (optional)

if [ -z "${RUN_ID:-}" ]; then
  latest="$(ls -t logs/*.txt 2>/dev/null | head -n 1 || true)"
  if [ -z "$latest" ]; then
    echo "error: RUN_ID is not set and no logs/*.txt found" >&2
    exit 1
  fi
  RUN_ID="$(basename "$latest" .txt)"
  echo "auto-detected RUN_ID: $RUN_ID"
fi

BRANCH="$(git branch --show-current)"
if [ -z "$BRANCH" ]; then
  echo "error: unable to detect current git branch" >&2
  exit 1
fi

if [ ! -f "logs/${RUN_ID}.txt" ]; then
  echo "error: missing required log file logs/${RUN_ID}.txt" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
else
  GPU_COUNT="0"
  GPU_NAME="unknown-gpu"
fi

GPU_SLUG="$(printf '%s' "$GPU_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g; s/-\+/-/g; s/^-//; s/-$//')"
DATE_TAG="$(date +%F)"
POD_NAME="$(hostname)"
RUN_DIR="vast/experiments/${DATE_TAG}-${GPU_SLUG}-${GPU_COUNT}gpu/${BRANCH}/${RUN_ID}"

mkdir -p "$RUN_DIR"

cp "logs/${RUN_ID}.txt" "$RUN_DIR/train.log"
cp "train_gpt.py" "$RUN_DIR/train_gpt.py"
[ -f "logs/telemetry/${RUN_ID}.csv" ] && cp "logs/telemetry/${RUN_ID}.csv" "$RUN_DIR/telemetry.csv" || echo "skipping missing: logs/telemetry/${RUN_ID}.csv"
[ -f "final_model.pt" ] && cp "final_model.pt" "$RUN_DIR/final_model.pt" || echo "skipping missing: final_model.pt"
[ -f "final_model.int8.ptz" ] && cp "final_model.int8.ptz" "$RUN_DIR/final_model.int8.ptz" || echo "skipping missing: final_model.int8.ptz"

FINAL_LINE="$(grep -E "final_int8_zlib_roundtrip_exact|final_int8_zlib_roundtrip" "logs/${RUN_ID}.txt" | tail -n 1 || true)"
HEAD_SHA="$(git rev-parse --short HEAD)"

cat > "$RUN_DIR/README.md" <<EOF
# ${RUN_ID}

- Branch: ${BRANCH}
- Commit: ${HEAD_SHA}
- Pod: ${POD_NAME}
- GPU: ${GPU_COUNT}x ${GPU_NAME}
- Run ID: \`${RUN_ID}\`

## Final Metric Line

\`\`\`
${FINAL_LINE}
\`\`\`
EOF

git add "$RUN_DIR"
git commit -m "logs: add ${RUN_ID} (${BRANCH}, ${GPU_SLUG}, ${GPU_COUNT}gpu)"
git push origin "$BRANCH"

echo "saved run -> ${RUN_DIR}"
