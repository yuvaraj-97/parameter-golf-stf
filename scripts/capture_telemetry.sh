#!/usr/bin/env bash
set -euo pipefail

# Capture periodic host/GPU telemetry during training.
# Usage:
#   RUN_ID=<run_id> ./scripts/capture_telemetry.sh
# Optional env:
#   TELEMETRY_INTERVAL_SECONDS (default: 15)
#   TELEMETRY_OUT_DIR (default: logs/telemetry)

if [ -z "${RUN_ID:-}" ]; then
  echo "error: RUN_ID must be set" >&2
  exit 1
fi

INTERVAL="${TELEMETRY_INTERVAL_SECONDS:-15}"
OUT_DIR="${TELEMETRY_OUT_DIR:-logs/telemetry}"
mkdir -p "$OUT_DIR"

OUT_FILE="${OUT_DIR}/${RUN_ID}.csv"
echo "timestamp_utc,gpu_index,gpu_name,util_gpu_pct,util_mem_pct,mem_used_mb,mem_total_mb,temp_c,power_w,sm_clock_mhz,mem_clock_mhz,cpu_util_pct,ram_used_mb,ram_total_mb,load_1m,load_5m,load_15m" > "$OUT_FILE"

cleanup() {
  echo "telemetry stopped: $OUT_FILE"
}
trap cleanup EXIT

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cpu_pct="$(top -bn1 | awk '/Cpu\(s\)/ {print 100 - $8; exit}')"
  mem_line="$(free -m | awk '/Mem:/ {print $3","$2; exit}')"
  load_line="$(awk '{print $1","$2","$3}' /proc/loadavg)"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits | while IFS=',' read -r idx name u_gpu u_mem m_used m_total temp power sm_clk mem_clk; do
      name="$(echo "$name" | xargs)"
      u_gpu="$(echo "$u_gpu" | xargs)"
      u_mem="$(echo "$u_mem" | xargs)"
      m_used="$(echo "$m_used" | xargs)"
      m_total="$(echo "$m_total" | xargs)"
      temp="$(echo "$temp" | xargs)"
      power="$(echo "$power" | xargs)"
      sm_clk="$(echo "$sm_clk" | xargs)"
      mem_clk="$(echo "$mem_clk" | xargs)"
      echo "${ts},${idx},${name},${u_gpu},${u_mem},${m_used},${m_total},${temp},${power},${sm_clk},${mem_clk},${cpu_pct},${mem_line},${load_line}" >> "$OUT_FILE"
    done
  else
    echo "${ts},-1,no-gpu,0,0,0,0,0,0,0,0,${cpu_pct},${mem_line},${load_line}" >> "$OUT_FILE"
  fi

  sleep "$INTERVAL"
done
