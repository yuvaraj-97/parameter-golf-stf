from __future__ import annotations

import csv
import html
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BRANCHES = [
    "baseline-repro",
    "stf-minimal",
    "stf-learned-gate",
    "stf-soft-freeze",
    "stf-reactivation",
    "stf-budget-regularization",
    "stf-recurrence",
    "stf-quantization",
]

BRANCH_MODEL = {
    "baseline-repro": {
        "family": "control",
        "decision": "none",
        "frozen_path": "none",
        "control": "none",
        "summary": "Control branch. No STF routing. Use this to judge whether STF helps at the same hardware and budget.",
    },
    "stf-minimal": {
        "family": "decision",
        "decision": "hard threshold",
        "frozen_path": "previous state",
        "control": "min active guardrail",
        "summary": "Plain STF: freeze low-change tokens and carry the previous hidden state forward.",
    },
    "stf-learned-gate": {
        "family": "decision",
        "decision": "learned sigmoid gate",
        "frozen_path": "continuous old/new blend",
        "control": "min active guardrail",
        "summary": "Learns the gate shape per layer instead of relying only on a fixed threshold.",
    },
    "stf-soft-freeze": {
        "family": "frozen path",
        "decision": "hard threshold",
        "frozen_path": "soft keep of new state",
        "control": "min active guardrail",
        "summary": "Keeps a fraction of the newly computed state for frozen tokens, reducing hard-freeze brittleness.",
    },
    "stf-reactivation": {
        "family": "control",
        "decision": "hard threshold",
        "frozen_path": "previous state",
        "control": "periodic reopen",
        "summary": "Periodically forces tokens active again so early freeze decisions are not permanent.",
    },
    "stf-budget-regularization": {
        "family": "control",
        "decision": "adaptive threshold",
        "frozen_path": "previous state",
        "control": "target active budget",
        "summary": "Adjusts threshold pressure to stay near a target active-token ratio.",
    },
    "stf-recurrence": {
        "family": "frozen path",
        "decision": "hard threshold",
        "frozen_path": "recurrent old/new blend",
        "control": "min active guardrail",
        "summary": "Uses a recurrence-like blend for frozen tokens instead of a pure old-state carry.",
    },
    "stf-quantization": {
        "family": "frozen path",
        "decision": "hard threshold",
        "frozen_path": "quantized previous state",
        "control": "min active guardrail",
        "summary": "Tests whether frozen tokens can carry a cheaper quantized state.",
    },
}

VAL_RE = re.compile(
    r"step:(\d+)/(\d+) val_loss:([0-9.]+) val_bpb:([0-9.]+).*?train_time:([0-9.]+)ms step_avg:([0-9.]+)ms"
)
TRAIN_RE = re.compile(
    r"step:(\d+)/(\d+) train_loss:([0-9.]+).*?train_time:([0-9.]+)ms step_avg:([0-9.]+)ms"
)
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip(?:_exact)? val_loss:([0-9.]+) val_bpb:([0-9.]+)")
MODE_RE = re.compile(r"stf_mode:([^\s]+)")
COMPILE_RE = re.compile(r"compile_model:([^\s]+)")
BATCH_RE = re.compile(r"train_batch_tokens:(\d+)")
SHARDS_RE = re.compile(r"train_loader:.*?train_shards:(\d+)")
TRAIN_SHARDS_RE = re.compile(r"train_loader:dataset:[^\s]+ train_shards:(\d+)")
MODEL_PARAMS_RE = re.compile(r"model_params:(\d+)")
CODE_BYTES_RE = re.compile(r"Code size: (\d+) bytes")
INT8_MODEL_RE = re.compile(
    r"Serialized model int8\+zlib: (\d+) bytes \(payload:(\d+) raw_torch:(\d+) payload_ratio:([0-9.]+)x\)"
)
INT8_TOTAL_RE = re.compile(r"Total submission size int8\+zlib: (\d+) bytes")
ARTIFACT_LIMIT_BYTES = 16_000_000


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")


def git_tree(ref: str) -> list[dict[str, str]]:
    result = run_git(["ls-tree", "-r", ref, "--", "runpod/experiments"])
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip() or "\t" not in line:
            continue
        meta, path = line.split("\t", 1)
        parts = meta.split()
        if len(parts) >= 3:
            rows.append({"sha": parts[2], "path": path})
    return rows


def git_blob(sha: str) -> str:
    result = run_git(["cat-file", "-p", sha])
    return result.stdout if result.returncode == 0 else ""


def parse_train_shards(text: str) -> int | None:
    match = SHARDS_RE.search(text)
    return int(match.group(1)) if match else None


def normalize_shard_filter(value: str | None) -> str:
    value = (value or "preferred").strip().lower()
    if value in {"", "auto", "preferred"}:
        return "preferred"
    if not value.isdigit():
        raise ValueError("shards must be 'preferred' or a numeric shard count")
    return value


def report_stem(iterations: str, shard_filter: str) -> str:
    return f"stf_{iterations}_branch_compare" if shard_filter == "preferred" else f"stf_{iterations}_shards{shard_filter}_branch_compare"


def choose_run(rows: list[dict[str, str]], iterations: str, shard_filter: str = "preferred") -> tuple[dict[str, str], str] | None:
    needle = f"i{iterations}"
    candidates = [row for row in rows if needle in row["path"] and row["path"].endswith("/train.log")]
    if not candidates:
        return None
    scored: list[tuple[tuple[object, ...], dict[str, str], str]] = []
    for row in candidates:
        text = git_blob(row["sha"])
        shards = parse_train_shards(text)
        if shard_filter != "preferred" and shards != int(shard_filter):
            continue
        shard_score = shards or 0
        scored.append(
            (
                (
                    shard_score == 80,
                    shard_score,
                    f"4gpu_i{iterations}_run1" in row["path"],
                    f"i{iterations}_run1" in row["path"],
                    row["path"],
                ),
                row,
                text,
            )
        )
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1], scored[0][2]


def sibling(rows: list[dict[str, str]], train_path: str, filename: str) -> dict[str, str] | None:
    wanted = train_path.rsplit("/", 1)[0] + "/" + filename
    return next((row for row in rows if row["path"] == wanted), None)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def parse_log(text: str) -> tuple[list[dict[str, float]], list[dict[str, float]], list[dict[str, float]], dict[str, object]]:
    vals: list[dict[str, float]] = []
    trains: list[dict[str, float]] = []
    finals: list[dict[str, float]] = []
    meta: dict[str, object] = {}

    for line in text.splitlines():
        if "stf_mode:" in line and "mode" not in meta:
            match = MODE_RE.search(line)
            if match:
                meta["mode"] = match.group(1)
        if "compile_model:" in line and "compile_model" not in meta:
            match = COMPILE_RE.search(line)
            if match:
                meta["compile_model"] = match.group(1)
        if "train_batch_tokens:" in line and "train_batch_tokens" not in meta:
            match = BATCH_RE.search(line)
            if match:
                meta["train_batch_tokens"] = int(match.group(1))
        if "train_loader:dataset:" in line and "train_shards" not in meta:
            match = TRAIN_SHARDS_RE.search(line)
            if match:
                meta["train_shards"] = int(match.group(1))
        if "model_params:" in line and "model_params" not in meta:
            match = MODEL_PARAMS_RE.search(line)
            if match:
                meta["model_params"] = int(match.group(1))
        if "Code size:" in line and "code_bytes" not in meta:
            match = CODE_BYTES_RE.search(line)
            if match:
                meta["code_bytes"] = int(match.group(1))
        if "Serialized model int8+zlib:" in line:
            match = INT8_MODEL_RE.search(line)
            if match:
                meta["int8_zlib_model_bytes"] = int(match.group(1))
                meta["int8_payload_bytes"] = int(match.group(2))
                meta["int8_raw_torch_bytes"] = int(match.group(3))
                meta["int8_payload_ratio"] = float(match.group(4))
        if "Total submission size int8+zlib:" in line:
            match = INT8_TOTAL_RE.search(line)
            if match:
                meta["total_submission_int8_zlib_bytes"] = int(match.group(1))

        match = VAL_RE.search(line)
        if match:
            vals.append(
                {
                    "step": int(match.group(1)),
                    "total": int(match.group(2)),
                    "val_loss": float(match.group(3)),
                    "val_bpb": float(match.group(4)),
                    "train_time_ms": float(match.group(5)),
                    "step_avg_ms": float(match.group(6)),
                }
            )
            continue

        match = TRAIN_RE.search(line)
        if match:
            trains.append(
                {
                    "step": int(match.group(1)),
                    "total": int(match.group(2)),
                    "train_loss": float(match.group(3)),
                    "train_time_ms": float(match.group(4)),
                    "step_avg_ms": float(match.group(5)),
                }
            )
            continue

        match = FINAL_RE.search(line)
        if match:
            finals.append({"val_loss": float(match.group(1)), "val_bpb": float(match.group(2))})

    return vals, trains, finals, meta


def parse_telemetry(text: str) -> list[dict[str, float]]:
    if not text:
        return []
    rows: list[dict[str, float]] = []
    for row in csv.DictReader(text.splitlines()):
        try:
            gpu_index = int(row.get("gpu_index") or -1)
            if gpu_index < 0:
                continue
            rows.append(
                {
                    "gpu_index": gpu_index,
                    "util_gpu_pct": float(row.get("util_gpu_pct") or 0),
                    "util_mem_pct": float(row.get("util_mem_pct") or 0),
                    "mem_used_mb": float(row.get("mem_used_mb") or 0),
                    "mem_total_mb": float(row.get("mem_total_mb") or 0),
                    "temp_c": float(row.get("temp_c") or 0),
                    "power_w": float(row.get("power_w") or 0),
                    "cpu_util_pct": float(row.get("cpu_util_pct") or 0),
                    "ram_used_mb": float(row.get("ram_used_mb") or 0),
                    "ram_total_mb": float(row.get("ram_total_mb") or 0),
                }
            )
        except ValueError:
            continue
    return rows


def fmt(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_bytes(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{int(value):,}"
    return str(value)


def collect(iterations: str, shard_filter: str = "preferred") -> dict[str, object]:
    records: list[dict[str, object]] = []
    curves: dict[str, list[dict[str, float]]] = {}
    missing: list[dict[str, str]] = []
    shard_filter = normalize_shard_filter(shard_filter)

    for branch in BRANCHES:
        rows = git_tree(f"origin/{branch}")
        chosen = choose_run(rows, iterations, shard_filter)
        model = BRANCH_MODEL.get(branch, {})
        if not chosen:
            records.append({"branch": branch, "status": "missing", **model})
            suffix = "" if shard_filter == "preferred" else f" with train_shards:{shard_filter}"
            missing.append({"branch": branch, "reason": f"No i{iterations}{suffix} train.log found on origin/{branch}"})
            continue

        run, log_text = chosen
        vals, trains, finals, meta = parse_log(log_text)
        telemetry_row = sibling(rows, run["path"], "telemetry.csv")
        telemetry = parse_telemetry(git_blob(telemetry_row["sha"]) if telemetry_row else "")

        util = [row["util_gpu_pct"] for row in telemetry]
        power = [row["power_w"] for row in telemetry]
        temp = [row["temp_c"] for row in telemetry]
        mem = [row["mem_used_mb"] for row in telemetry]
        mem_total = max([row["mem_total_mb"] for row in telemetry] or [0])
        best = min(vals, key=lambda value: value["val_bpb"]) if vals else None
        final = vals[-1] if vals else None
        roundtrip = finals[-1] if finals else None
        latest_train = trains[-1] if trains else None
        total_artifact_bytes = meta.get("total_submission_int8_zlib_bytes", "")
        artifact_margin_bytes = (
            ARTIFACT_LIMIT_BYTES - total_artifact_bytes
            if isinstance(total_artifact_bytes, int)
            else ""
        )

        record = {
            "branch": branch,
            "status": "ok" if vals else "no_val_points",
            "path": run["path"],
            "mode": meta.get("mode", "baseline" if branch == "baseline-repro" else ""),
            "compile_model": meta.get("compile_model", ""),
            "train_batch_tokens": meta.get("train_batch_tokens", ""),
            "train_shards": meta.get("train_shards", parse_train_shards(log_text) or ""),
            "model_params": meta.get("model_params", ""),
            "val_points": len(vals),
            "best_step": best["step"] if best else "",
            "best_val_bpb": best["val_bpb"] if best else "",
            "final_step": final["step"] if final else "",
            "final_val_bpb": final["val_bpb"] if final else "",
            "roundtrip_val_bpb": roundtrip["val_bpb"] if roundtrip else "",
            "int8_zlib_model_bytes": meta.get("int8_zlib_model_bytes", ""),
            "int8_payload_bytes": meta.get("int8_payload_bytes", ""),
            "int8_raw_torch_bytes": meta.get("int8_raw_torch_bytes", ""),
            "int8_payload_ratio": meta.get("int8_payload_ratio", ""),
            "code_bytes": meta.get("code_bytes", ""),
            "total_submission_int8_zlib_bytes": total_artifact_bytes,
            "artifact_margin_bytes": artifact_margin_bytes,
            "step_avg_ms": final["step_avg_ms"] if final else (latest_train["step_avg_ms"] if latest_train else ""),
            "telemetry_samples": len(telemetry),
            "gpu_util_avg": mean(util),
            "gpu_util_p10": percentile(util, 0.10),
            "gpu_util_p50": percentile(util, 0.50),
            "gpu_util_p90": percentile(util, 0.90),
            "gpu_util_lt80_frac": (sum(1 for value in util if value < 80) / len(util)) if util else None,
            "power_avg_w": mean(power),
            "temp_avg_c": mean(temp),
            "mem_avg_gb": mean(mem) / 1024 if mem else None,
            "mem_peak_gb": max(mem) / 1024 if mem else None,
            "mem_total_gb": mem_total / 1024 if mem_total else None,
            **model,
        }
        records.append(record)
        curves[branch] = vals

    return {
        "iterations": iterations,
        "shardFilter": shard_filter,
        "records": records,
        "curves": curves,
        "missing": missing,
        "branchModel": BRANCH_MODEL,
    }


def write_csv(data: dict[str, object], path: Path) -> None:
    fields = [
        "branch",
        "status",
        "family",
        "decision",
        "frozen_path",
        "control",
        "mode",
        "compile_model",
        "train_batch_tokens",
        "train_shards",
        "model_params",
        "val_points",
        "best_step",
        "best_val_bpb",
        "final_step",
        "final_val_bpb",
        "roundtrip_val_bpb",
        "int8_zlib_model_bytes",
        "int8_payload_bytes",
        "int8_raw_torch_bytes",
        "int8_payload_ratio",
        "code_bytes",
        "total_submission_int8_zlib_bytes",
        "artifact_margin_bytes",
        "step_avg_ms",
        "telemetry_samples",
        "gpu_util_avg",
        "gpu_util_p10",
        "gpu_util_p50",
        "gpu_util_p90",
        "gpu_util_lt80_frac",
        "power_avg_w",
        "temp_avg_c",
        "mem_avg_gb",
        "mem_peak_gb",
        "mem_total_gb",
        "path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in data["records"]:
            writer.writerow({field: record.get(field, "") for field in fields})


def write_html(data: dict[str, object], path: Path) -> None:
    ok = [record for record in data["records"] if record.get("status") == "ok"]
    def ranking_score(record: dict[str, object]) -> float:
        value = record.get("roundtrip_val_bpb") or record.get("final_val_bpb")
        return float(value) if isinstance(value, (int, float)) else float("inf")

    ok_sorted = sorted(ok, key=ranking_score)
    iterations = str(data["iterations"])
    shard_filter = normalize_shard_filter(str(data.get("shardFilter", "preferred")))
    shard_label = "prefer 80 / highest available" if shard_filter == "preferred" else f"exactly {shard_filter}"
    parsed_shards = sorted({str(record.get("train_shards")) for record in ok if record.get("train_shards")})
    parsed_shards_label = ", ".join(parsed_shards) if parsed_shards else "n/a"
    missing_html = "".join(
        f"<li><b>{html.escape(item['branch'])}</b>: {html.escape(item['reason'])}</li>"
        for item in data["missing"]
    )
    table_rows = "".join(
        f"<tr><td>{idx}</td><td><b>{html.escape(record['branch'])}</b><br><span>{html.escape(str(record.get('summary','')))}</span></td>"
        f"<td><b>{fmt(record.get('roundtrip_val_bpb'))}</b><br><span>raw {fmt(record.get('final_val_bpb'))}</span></td>"
        f"<td>{fmt(record.get('best_val_bpb'))} @ {fmt(record.get('best_step'), 0)}</td>"
        f"<td>{fmt_bytes(record.get('total_submission_int8_zlib_bytes'))}<br><span>margin {fmt_bytes(record.get('artifact_margin_bytes'))}</span></td>"
        f"<td>{fmt(record.get('train_shards'), 0)}</td><td>{html.escape(str(record.get('family','')))}</td>"
        f"<td>{fmt(record.get('step_avg_ms'), 2)} ms</td><td>{fmt(record.get('gpu_util_avg'), 1)}%</td>"
        f"<td>{fmt(record.get('mem_peak_gb'), 2)} GB</td></tr>"
        for idx, record in enumerate(ok_sorted, 1)
    )
    scale_options = "".join(
        f'<option value="{scale}" {"selected" if scale == iterations else ""}>{label}</option>'
        for scale, label in [
            ("2000", "2k"),
            ("5000", "5k"),
            ("10000", "10k"),
            ("20000", "20k"),
        ]
    )
    shard_options = "".join(
        f'<option value="{value}" {"selected" if value == shard_filter else ""}>{label}</option>'
        for value, label in [
            ("preferred", "Prefer 80 shards / highest available"),
            ("80", "Exactly 80 shards"),
            ("1", "Exactly 1 shard"),
        ]
    )
    payload = json.dumps(data).replace("</", "<\\/")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    document = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STF Branch Comparison</title>
  <style>
    :root { --bg:#080b10; --panel:#101722; --panel2:#151f2d; --ink:#eef4ff; --muted:#91a0b8; --faint:#5e6b80; --line:#263448; --accent:#ff8a4c; --cyan:#58d7e8; --green:#8ee06d; --gold:#ffc857; --red:#ff6b6b; --violet:#b99cff; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at 8% -8%, #1f4c58 0, transparent 29rem), radial-gradient(circle at 90% 4%, #5b3728 0, transparent 24rem), linear-gradient(160deg,#05070b,#0d1420 48%,#111827); color:var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
    main { max-width:1480px; margin:auto; padding:28px 20px 52px; }
    section { background:linear-gradient(180deg, rgba(21,31,45,.94), rgba(12,18,27,.94)); border:1px solid var(--line); border-radius:24px; box-shadow:0 24px 72px #0009; padding:22px; margin:18px 0; }
    h1 { font-size:clamp(2.35rem,4.6vw,5rem); line-height:.92; letter-spacing:-.065em; margin:0 0 14px; }
    h2 { margin:0 0 14px; font-size:1.28rem; letter-spacing:-.02em; }
    h3 { margin:0 0 10px; font-size:1rem; }
    p, li { color:var(--muted); line-height:1.58; }
    a { color:var(--cyan); text-decoration:none; }
    code { background:#0a1018; border:1px solid var(--line); border-radius:6px; padding:1px 6px; color:#dce8ff; }
    table { width:100%; border-collapse:collapse; font-size:.9rem; }
    th,td { border-bottom:1px solid var(--line); padding:10px 8px; text-align:left; vertical-align:top; }
    th { color:#9fb0c8; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; }
    td span { color:var(--muted); font-size:.8rem; }
    select { width:100%; border:1px solid var(--line); border-radius:13px; background:#0b111a; color:var(--ink); padding:11px 12px; font-size:1rem; outline:none; }
    select:focus { border-color:var(--cyan); box-shadow:0 0 0 3px #58d7e822; }
    input { width:100%; border:1px solid var(--line); border-radius:13px; background:#0b111a; color:var(--ink); padding:11px 12px; font-size:1rem; outline:none; }
    button { border:1px solid var(--cyan); border-radius:13px; background:var(--cyan); color:#081018; padding:11px 14px; font-size:1rem; font-weight:800; cursor:pointer; }
    button:hover { filter:brightness(1.08); }
    .hero { display:grid; grid-template-columns:1.25fr .75fr; gap:18px; align-items:stretch; }
    .eyebrow { text-transform:uppercase; letter-spacing:.13em; color:var(--accent); margin:0 0 10px; font-weight:700; }
    .scale-row { display:grid; grid-template-columns:140px 1fr auto; gap:9px; margin-top:14px; align-items:center; }
    .scale-label { color:var(--muted); font-size:.82rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }
    .scale-pill { border:1px solid var(--line); border-radius:999px; padding:7px 12px; background:#0b111a; color:var(--muted); font-weight:700; }
    .scale-pill.active { color:#081018; background:var(--cyan); border-color:var(--cyan); }
    .kpis { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .kpi { background:linear-gradient(180deg,#0e1520,#0a1018); border:1px solid var(--line); border-radius:17px; padding:14px; }
    .kpi span { display:block; color:var(--muted); font-size:.82rem; }
    .kpi b { display:block; margin-top:5px; font-size:1.35rem; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .full { grid-column:1 / -1; }
    .chart-wrap { position:relative; }
    canvas { width:100%; height:370px; background:linear-gradient(180deg,#0b111a,#090e15); border:1px solid var(--line); border-radius:18px; cursor:crosshair; }
    .compare-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .select-row { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:16px; }
    .branch-card { border:1px solid var(--line); background:linear-gradient(180deg,#0e1520,#0a1018); border-radius:18px; padding:16px; }
    .metric-row { display:grid; grid-template-columns:1fr auto; gap:8px; border-top:1px solid var(--line); padding-top:8px; margin-top:8px; }
    .metric-row span { color:var(--muted); }
    .delta-good { color:var(--green); font-weight:800; }
    .delta-bad { color:var(--red); font-weight:800; }
    .pipeline { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:12px; }
    .node { min-height:104px; border:1px solid var(--line); border-radius:16px; background:#0b111a; padding:13px; position:relative; overflow:hidden; }
    .node::before { content:""; position:absolute; inset:0 0 auto 0; height:5px; background:var(--node); }
    .node small { display:block; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-size:.68rem; margin-bottom:7px; }
    .node b { display:block; }
    .plain-explainer { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin:12px 0 16px; }
    .plain-card { border:1px solid var(--line); background:#0b111a; border-radius:16px; padding:14px; }
    .plain-card b { display:block; margin-bottom:6px; }
    .plain-card span { color:var(--muted); line-height:1.45; font-size:.9rem; }
    .token-panel { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .token-lane { border:1px solid var(--line); background:#0b111a; border-radius:18px; padding:15px; }
    .route-diagram { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin:12px 0; align-items:stretch; }
    .route-step { border:1px solid #314157; background:#080d14; border-radius:14px; padding:12px; min-height:82px; position:relative; }
    .route-step small { display:block; color:var(--faint); text-transform:uppercase; letter-spacing:.08em; font-size:.66rem; margin-bottom:5px; }
    .route-step b { display:block; }
    .route-step p { margin:.35rem 0 0; font-size:.84rem; line-height:1.35; }
    .route-wide { grid-column:1 / -1; }
    .lane-box { border-radius:14px; padding:12px; min-height:92px; }
    .lane-update { border:1px solid #487a45; background:linear-gradient(180deg,#19361f,#101c14); }
    .lane-reuse { border:1px solid #4a596f; background:linear-gradient(180deg,#202a38,#101720); }
    .lane-disabled { opacity:.42; filter:saturate(.55); }
    .route-arrow { color:var(--faint); text-align:center; font-size:1.1rem; margin:-2px 0; }
    .simple-badge { display:inline-flex; border-radius:999px; padding:5px 8px; margin:2px 4px 2px 0; background:#111a26; border:1px solid var(--line); color:#cbd7ea; font-size:.76rem; }
    .tokens { display:grid; grid-template-columns:repeat(12, minmax(0,1fr)); gap:7px; margin:12px 0; }
    .token { height:28px; border-radius:9px; border:1px solid #ffffff12; position:relative; overflow:hidden; }
    .token::after { content:attr(data-i); position:absolute; inset:auto 4px 3px auto; color:#071018aa; font-size:.58rem; font-weight:800; }
    .token.updated { background:var(--green); }
    .token.frozen { background:#38465a; }
    .token.blend { background:linear-gradient(90deg,var(--green),var(--cyan)); }
    .token.reopen { background:linear-gradient(135deg,var(--gold),var(--green)); }
    .token.quant { background:repeating-linear-gradient(135deg,#7f8ba0 0 5px,#4b596d 5px 10px); }
    .token.recur { background:linear-gradient(135deg,var(--violet),#5263ff); }
    .token-copy { color:var(--muted); font-size:.86rem; min-height:42px; }
    .legend { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    .pill { display:inline-flex; gap:7px; align-items:center; border:1px solid var(--line); border-radius:999px; padding:6px 10px; background:#0b111a; color:#cbd7ea; font-size:.84rem; }
    .dot { width:10px; height:10px; border-radius:50%; background:var(--c); }
    .warn { background:#201a0d; border-color:#6d5521; }
    .tooltip { pointer-events:none; position:fixed; z-index:20; transform:translate(14px,14px); min-width:190px; max-width:300px; background:#05080dcc; backdrop-filter:blur(8px); border:1px solid #405069; border-radius:13px; padding:10px 11px; box-shadow:0 18px 42px #000b; display:none; color:var(--ink); font-size:.84rem; }
    .tooltip b { display:block; margin-bottom:5px; }
    .tooltip span { display:block; color:var(--muted); margin-top:2px; }
    .hint { color:var(--faint); font-size:.83rem; margin-top:8px; }
    @media(max-width:980px) { .hero,.grid,.compare-grid,.select-row,.pipeline,.token-panel,.scale-row,.plain-explainer,.route-diagram { grid-template-columns:1fr; } .route-wide { grid-column:auto; } }
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <p class="eyebrow">STF branch analysis</p>
      <h1>__ITERATIONS__-iteration branch comparison</h1>
      <p>This report is generated from git-archived <code>train.log</code> and <code>telemetry.csv</code>. Re-run <code>python scripts/generate_stf_analysis.py __ITERATIONS__ __SHARD_FILTER__</code> after pulling logs to refresh the page from the active branch files.</p>
      <p><b>Generated:</b> __GENERATED_AT__ &nbsp; <b>Source:</b> local remote-tracking refs</p>
      <div class="scale-row" aria-label="iteration reports">
        <span class="scale-label">Iteration</span>
        <select id="iterationSelect">__SCALE_OPTIONS__</select>
        <button id="loadIteration" type="button">Load / generate</button>
      </div>
      <div class="scale-row">
        <span class="scale-label">Shards</span>
        <select id="shardSelect">__SHARD_OPTIONS__</select>
        <span class="hint">Default prefers 80-shard runs when both 1 and 80 exist.</span>
      </div>
      <div class="scale-row" style="grid-template-columns:140px 1fr auto;">
        <span class="scale-label">Custom</span>
        <input id="customIterations" inputmode="numeric" pattern="[0-9]*" placeholder="optional, e.g. 50000">
        <button id="loadCustomIteration" type="button">Use custom</button>
      </div>
      <p class="hint" id="scaleStatus">If this page is served by <code>python scripts/serve_stf_analysis.py</code>, missing reports are generated automatically. In plain file mode, the page will show the command to run.</p>
    </div>
    <div class="kpis">
      <div class="kpi"><span>Parsed runs</span><b>__OK_COUNT__</b></div>
      <div class="kpi"><span>Missing</span><b>__MISSING_COUNT__</b></div>
      <div class="kpi"><span>Best parsed</span><b>__BEST_BRANCH__</b></div>
      <div class="kpi"><span>Final BPB</span><b>__BEST_BPB__</b></div>
      <div class="kpi"><span>Shard mode</span><b>__SHARD_LABEL__</b></div>
      <div class="kpi"><span>Parsed shards</span><b>__PARSED_SHARDS__</b></div>
    </div>
  </section>
  __MISSING_SECTION__
  <section>
    <h2>Two-branch comparator</h2>
    <div class="select-row">
      <select id="branchA"></select>
      <select id="branchB"></select>
    </div>
    <div class="compare-grid">
      <div class="branch-card" id="cardA"></div>
      <div class="branch-card" id="cardB"></div>
    </div>
    <h2 style="margin-top:18px;">How STF changes token flow</h2>
    <div class="plain-explainer">
      <div class="plain-card"><b>Token</b><span>A small piece of text. The numbered blocks below are example tokens moving through the model.</span></div>
      <div class="plain-card"><b>Layer</b><span>One repeated thinking step inside the model. A small model still runs many layers for every token.</span></div>
      <div class="plain-card"><b>Hidden state</b><span>The model's working note for a token. If the note is barely changing, STF may reuse it.</span></div>
      <div class="plain-card"><b>Update lane</b><span>The normal expensive path. A token gets fresh main-model math instead of reusing old work.</span></div>
      <div class="plain-card"><b>Reuse lane</b><span>The shortcut path. A stable token keeps or blends an older working note instead of full compute.</span></div>
      <div class="plain-card"><b>What good means</b><span>Final BPB stays close to baseline while step time, GPU work, or memory pressure improves.</span></div>
    </div>
    <p class="hint">This is a teaching visual, not the exact frozen-token list from the run. Current logs do not save which individual tokens froze at each layer.</p>
    <div class="token-panel" id="tokenFlow"></div>
    <div class="pipeline" id="pipeline"></div>
    <div class="chart-wrap"><canvas id="pairCurve" style="margin-top:16px;"></canvas></div>
  </section>
  <section>
    <h2>All-branch ranking</h2>
    <table><thead><tr><th>#</th><th>Branch</th><th>Roundtrip BPB</th><th>Best raw BPB</th><th>Artifact bytes</th><th>Shards</th><th>Family</th><th>Step avg</th><th>GPU avg</th><th>VRAM peak</th></tr></thead><tbody>__TABLE_ROWS__</tbody></table>
  </section>
  <section class="grid">
    <div>
      <h2>All validation curves</h2>
      <div class="chart-wrap"><canvas id="allCurve"></canvas></div>
      <div class="legend" id="legend"></div>
      <p class="hint">Move the cursor over a point to inspect branch, step, BPB, and step average.</p>
    </div>
    <div>
      <h2>Roundtrip BPB bars</h2>
      <div class="chart-wrap"><canvas id="bars"></canvas></div>
      <p class="hint">Hover bars for roundtrip BPB, artifact size, GPU utilization, and VRAM peak.</p>
    </div>
  </section>
</main>
<div class="tooltip" id="tooltip"></div>
<script>
const DATA = __PAYLOAD__;
const colors = ['#b84f31','#246b70','#527b39','#8a5a9b','#a97828','#5d6f91','#9d3f5f','#3d7d5d'];
const chartState = {};
const okRecords = DATA.records.filter(r => r.status === 'ok');
const byBranch = Object.fromEntries(DATA.records.map(r => [r.branch, r]));
function f(v, d=4) { return Number.isFinite(v) ? v.toFixed(d) : 'n/a'; }
function fb(v) { return Number.isFinite(v) ? Math.trunc(v).toLocaleString() : 'n/a'; }
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function setup(c) { const dpr=window.devicePixelRatio||1, r=c.getBoundingClientRect(); c.width=r.width*dpr; c.height=r.height*dpr; const ctx=c.getContext('2d'); ctx.scale(dpr,dpr); return [ctx,r.width,r.height]; }
function axes(ctx,w,h,p,ymin,ymax,xmax) {
  ctx.strokeStyle='#263448'; ctx.lineWidth=1; ctx.font='12px sans-serif';
  for(let i=0;i<=4;i++){ const y=p+(h-2*p)*i/4; ctx.beginPath(); ctx.moveTo(p,y); ctx.lineTo(w-p,y); ctx.stroke(); }
  ctx.strokeStyle='#46566d'; ctx.beginPath(); ctx.moveTo(p,h-p); ctx.lineTo(w-p,h-p); ctx.lineTo(w-p,p); ctx.stroke();
  ctx.fillStyle='#91a0b8'; ctx.fillText('BPB ' + ymax.toFixed(3), p, p-12); ctx.fillText(ymin.toFixed(3), w-p-52, h-p+18); ctx.fillText('step ' + xmax, w-p-58, h-p+34);
}
function drawCurve(id, branches) {
  const [ctx,w,h]=setup(document.getElementById(id)), p=42; ctx.clearRect(0,0,w,h);
  const pts=branches.flatMap(b => DATA.curves[b] || []); if(!pts.length) return;
  const xs=pts.map(x=>x.step), ys=pts.map(x=>x.val_bpb), xmin=Math.min(...xs), xmax=Math.max(...xs), rawMin=Math.min(...ys), rawMax=Math.max(...ys);
  const pad=Math.max((rawMax-rawMin)*0.08, 0.01), ymin=rawMin-pad, ymax=rawMax+pad;
  axes(ctx,w,h,p,ymin,ymax,xmax);
  const X=s=>p+(s-xmin)/(xmax-xmin||1)*(w-2*p), Y=v=>h-p-(v-ymin)/(ymax-ymin||1)*(h-2*p);
  const hit=[];
  branches.forEach((b,i)=>{ const arr=DATA.curves[b]||[], color=colors[i%colors.length]; ctx.strokeStyle=color; ctx.lineWidth=2.4; ctx.beginPath(); arr.forEach((pt,j)=>{ const x=X(pt.step), y=Y(pt.val_bpb); if(j) ctx.lineTo(x,y); else ctx.moveTo(x,y); hit.push({type:'curve', x, y, branch:b, color, ...pt}); }); ctx.stroke(); arr.forEach(pt=>{ const x=X(pt.step), y=Y(pt.val_bpb); ctx.fillStyle=color; ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2); ctx.fill(); }); });
  chartState[id]={type:'curve', hit};
}
function drawBars() {
  const score = r => Number.isFinite(r.roundtrip_val_bpb) ? r.roundtrip_val_bpb : r.final_val_bpb;
  const [ctx,w,h]=setup(document.getElementById('bars')), p=35; const rows=[...okRecords].sort((a,b)=>score(a)-score(b));
  ctx.clearRect(0,0,w,h); const mx=Math.max(...rows.map(score)), mn=Math.min(...rows.map(score)); const gap=(h-2*p)/Math.max(rows.length,1), bh=Math.min(gap*.62, 28); ctx.font='12px sans-serif';
  const hit=[];
  rows.forEach((r,i)=>{ const v=score(r), y=p+i*gap, ww=(v-mn)/(mx-mn||1)*(w-270)+28; ctx.fillStyle=colors[i%colors.length]; ctx.fillRect(190,y,ww,bh); ctx.fillStyle='#dce8ff'; ctx.fillText(r.branch,8,y+bh*.72); ctx.fillText(f(v),200+ww,y+bh*.72); hit.push({type:'bar', x:190, y, w:ww, h:bh, record:r, color:colors[i%colors.length]}); });
  chartState.bars={type:'bar', hit};
}
function metric(label, a, b, lower=true, suffix='') {
  const delta = Number.isFinite(a) && Number.isFinite(b) ? b - a : NaN;
  const cls = !Number.isFinite(delta) ? '' : ((lower ? delta < 0 : delta > 0) ? 'delta-good' : 'delta-bad');
  return `<div class="metric-row"><span>${label}</span><b>${f(a, label.includes('ms') ? 2 : 4)}${suffix}</b></div><div class="metric-row"><span>vs other</span><b class="${cls}">${Number.isFinite(delta) ? (delta > 0 ? '+' : '') + delta.toFixed(label.includes('ms') ? 2 : 4) + suffix : 'n/a'}</b></div>`;
}
function card(r, other) {
  return `<h3>${r.branch}</h3><p>${r.summary || ''}</p>
  <div class="metric-row"><span>family</span><b>${r.family || ''}</b></div>
  <div class="metric-row"><span>decision</span><b>${r.decision || ''}</b></div>
  <div class="metric-row"><span>frozen path</span><b>${r.frozen_path || ''}</b></div>
  <div class="metric-row"><span>control</span><b>${r.control || ''}</b></div>
  <div class="metric-row"><span>train shards</span><b>${r.train_shards || 'n/a'}</b></div>
  <div class="metric-row"><span>roundtrip BPB</span><b>${f(r.roundtrip_val_bpb)}</b></div>
  <div class="metric-row"><span>artifact bytes</span><b>${fb(r.total_submission_int8_zlib_bytes)}</b></div>
  <div class="metric-row"><span>16MB margin</span><b class="${Number.isFinite(r.artifact_margin_bytes) && r.artifact_margin_bytes >= 0 ? 'delta-good' : 'delta-bad'}">${fb(r.artifact_margin_bytes)}</b></div>
  ${metric('final BPB', r.final_val_bpb, other.final_val_bpb)}
  ${metric('step avg ms', r.step_avg_ms, other.step_avg_ms)}
  ${metric('GPU util', r.gpu_util_avg, other.gpu_util_avg, false, '%')}`;
}
function tokenPattern(r) {
  const mode = r.branch || '';
  const out = [];
  for(let i=0;i<24;i++){
    let cls='updated', label='full transformer update';
    if(mode === 'baseline-repro') { cls='updated'; label='always updated'; }
    else if(mode.includes('learned-gate')) { cls=i%3===0?'updated':'blend'; label='learned old/new gate'; }
    else if(mode.includes('soft-freeze')) { cls=i%4===0?'updated':'blend'; label='soft keep of new state'; }
    else if(mode.includes('reactivation')) { cls=i%5===0?'reopen':(i%2?'frozen':'updated'); label='periodic reopen plus freeze'; }
    else if(mode.includes('budget')) { cls=i%3===0?'updated':'frozen'; label='threshold adjusted to budget'; }
    else if(mode.includes('recurrence')) { cls=i%4===0?'updated':'recur'; label='recurrent old/new blend'; }
    else if(mode.includes('quantization')) { cls=i%4===0?'updated':'quant'; label='quantized carry state'; }
    else { cls=i%4===0?'updated':'frozen'; label='hard threshold freeze'; }
    out.push(`<span class="token ${cls}" data-i="${i+1}" title="${esc(label)}"></span>`);
  }
  return out.join('');
}
function tokenCopy(r) {
  if(r.branch === 'baseline-repro') return 'Plain English: every token pays the full compute cost at every layer. There is no shortcut lane.';
  if(r.branch.includes('learned-gate')) return 'Plain English: the model learns how much of the new update to mix with the old state instead of using only a fixed rule.';
  if(r.branch.includes('soft-freeze')) return 'Plain English: even frozen tokens keep a small part of the new update, so freezing is less abrupt.';
  if(r.branch.includes('reactivation')) return 'Plain English: frozen tokens are periodically reopened so an early freeze decision does not trap them forever.';
  if(r.branch.includes('budget')) return 'Plain English: the branch tries to keep the number of fully-updated tokens near a target budget.';
  if(r.branch.includes('recurrence')) return 'Plain English: frozen tokens use a recurrent blend of old and new state rather than a pure copy.';
  if(r.branch.includes('quantization')) return 'Plain English: frozen tokens reuse a cheaper compressed version of their previous state.';
  return 'Plain English: stable tokens skip the expensive update and reuse their previous state.';
}
function routeBadges(r) {
  if(r.branch === 'baseline-repro') return ['all tokens update', 'highest compute', 'control branch'];
  return [`decision: ${r.decision}`, `reuse lane: ${r.frozen_path}`, `control: ${r.control}`];
}
function routeDiagram(r, idx) {
  const baseline = r.branch === 'baseline-repro';
  const badges = routeBadges(r).map(x => `<span class="simple-badge">${esc(x)}</span>`).join('');
  const reuseClass = baseline ? 'lane-box lane-reuse lane-disabled' : 'lane-box lane-reuse';
  const reuseText = baseline ? 'Disabled in baseline. No token is allowed to skip.' : `Stable tokens go here. They reuse ${esc(r.frozen_path || 'previous state')}.`;
  return `<div class="token-lane">
    <h3>${idx===0?'A':'B'}: ${esc(r.branch)}</h3>
    <div class="route-diagram">
      <div class="route-step route-wide"><small>1. Text pieces enter</small><b>Tokens arrive at a layer</b><p>Each numbered block is one token being processed by this branch.</p><div class="tokens">${tokenPattern(r)}</div></div>
      <div class="route-step"><small>2. Change check</small><b>Is this token still changing?</b><p>STF compares the token's current working note with its older note.</p></div>
      <div class="route-step"><small>3. Branch rule</small><b>${esc(r.decision || 'no STF rule')}</b><p>${baseline ? 'Baseline does not make a routing decision.' : 'This branch decides which lane each token takes.'}</p></div>
      <div class="lane-box lane-update"><small>Update lane</small><b>Run main math</b><p>Important or changing tokens get a fresh full update.</p></div>
      <div class="${reuseClass}"><small>Reuse lane</small><b>${baseline ? 'Not used' : 'Freeze / carry'}</b><p>${reuseText}</p></div>
    </div>
    <div>${badges}</div>
    <div class="token-copy">${esc(tokenCopy(r))}</div>
  </div>`;
}
function renderTokenFlow(a,b) {
  document.getElementById('tokenFlow').innerHTML = [a,b].map(routeDiagram).join('');
}
function pipeline(a, b) {
  const steps = [
    ['Decision', a.decision, b.decision, '#b84f31'],
    ['Freeze signal', 'EMA(||h_new - h_old||)', 'EMA(||h_new - h_old||)', '#246b70'],
    ['Frozen path', a.frozen_path, b.frozen_path, '#527b39'],
    ['Control', a.control, b.control, '#a97828'],
  ];
  document.getElementById('pipeline').innerHTML = steps.map(s => `<div class="node" style="--node:${s[3]}"><small>${s[0]}</small><b>A: ${s[1]}</b><p style="margin:.4rem 0 0;">B: ${s[2]}</p></div>`).join('');
}
function renderPair() {
  const a=byBranch[document.getElementById('branchA').value], b=byBranch[document.getElementById('branchB').value];
  document.getElementById('cardA').innerHTML=card(a,b); document.getElementById('cardB').innerHTML=card(b,a); renderTokenFlow(a,b); pipeline(a,b); drawCurve('pairCurve',[a.branch,b.branch]);
}
function selectedIteration() {
  const custom = document.getElementById('customIterations')?.value.trim();
  return custom || document.getElementById('iterationSelect').value;
}
function selectedShard() {
  return document.getElementById('shardSelect')?.value || 'preferred';
}
function reportName(iter, shard) {
  return shard === 'preferred' ? `stf_${iter}_branch_compare.html` : `stf_${iter}_shards${shard}_branch_compare.html`;
}
function setScaleStatus(html) {
  document.getElementById('scaleStatus').innerHTML = html;
}
async function loadIteration() {
  const iter = selectedIteration();
  const shard = selectedShard();
  if(!/^\\d+$/.test(iter)) { setScaleStatus('Iteration must be numeric.'); return; }
  if(iter === String(DATA.iterations) && shard === String(DATA.shardFilter || 'preferred')) { setScaleStatus('Already viewing this iteration and shard report.'); return; }
  const target = reportName(iter, shard);
  const command = `python scripts/generate_stf_analysis.py ${iter} ${shard}`;
  const serverMode = location.protocol === 'http:' || location.protocol === 'https:';
  if(!serverMode) {
    setScaleStatus(`Static file mode cannot execute Python. Run <code>${command}</code>, then open <a href="${target}">${target}</a>.`);
    return;
  }
  setScaleStatus(`Generating/loading <code>${iter}</code> with shard mode <code>${shard}</code> from branch logs...`);
  try {
    const res = await fetch(`/api/report?iterations=${encodeURIComponent(iter)}&shards=${encodeURIComponent(shard)}`);
    const body = await res.json().catch(() => ({}));
    if(!res.ok || !body.ok) {
      setScaleStatus(`Generation failed. Run <code>${command}</code> in the repo root. ${esc(body.error || '')}`);
      return;
    }
    window.location.href = target;
  } catch (err) {
    setScaleStatus(`Could not reach the local analysis server. Run <code>python scripts/serve_stf_analysis.py</code> or run <code>${command}</code> manually.`);
  }
}
function tooltipHtml(item) {
  if(item.type === 'bar') { const r=item.record; return `<b>${esc(r.branch)}</b><span>train shards: ${r.train_shards || 'n/a'}</span><span>roundtrip BPB: ${f(r.roundtrip_val_bpb)}</span><span>final raw BPB: ${f(r.final_val_bpb)}</span><span>best BPB: ${f(r.best_val_bpb)} @ step ${r.best_step}</span><span>artifact: ${fb(r.total_submission_int8_zlib_bytes)} bytes</span><span>16MB margin: ${fb(r.artifact_margin_bytes)} bytes</span><span>step avg: ${f(r.step_avg_ms,2)} ms</span><span>GPU avg: ${f(r.gpu_util_avg,1)}%</span><span>VRAM peak: ${f(r.mem_peak_gb,2)} GB</span>`; }
  return `<b>${esc(item.branch)}</b><span>step: ${item.step} / ${item.total}</span><span>val BPB: ${f(item.val_bpb)}</span><span>val loss: ${f(item.val_loss)}</span><span>step avg: ${f(item.step_avg_ms,2)} ms</span>`;
}
function attachHover(canvas) {
  const tip=document.getElementById('tooltip');
  canvas.addEventListener('mousemove', ev => {
    const state=chartState[canvas.id]; if(!state) return;
    const r=canvas.getBoundingClientRect(), x=ev.clientX-r.left, y=ev.clientY-r.top;
    let item=null;
    if(state.type === 'bar') item=state.hit.find(h => x>=h.x && x<=h.x+h.w && y>=h.y && y<=h.y+h.h);
    else {
      let best={d:999,item:null};
      state.hit.forEach(h => { const d=Math.hypot(h.x-x,h.y-y); if(d<best.d) best={d,item:h}; });
      if(best.d < 26) item=best.item;
    }
    if(!item) { tip.style.display='none'; return; }
    tip.innerHTML=tooltipHtml(item); tip.style.left=ev.clientX+'px'; tip.style.top=ev.clientY+'px'; tip.style.display='block';
  });
  canvas.addEventListener('mouseleave', () => { tip.style.display='none'; });
}
function init() {
  const opts = okRecords.map(r => `<option value="${r.branch}">${r.branch}</option>`).join('');
  branchA.innerHTML=opts; branchB.innerHTML=opts; branchA.value=okRecords[0]?.branch || ''; branchB.value=okRecords[1]?.branch || branchA.value;
  branchA.onchange=renderPair; branchB.onchange=renderPair; renderPair();
  document.getElementById('loadIteration').onclick=loadIteration;
  document.getElementById('loadCustomIteration').onclick=loadIteration;
  const all=okRecords.map(r=>r.branch); drawCurve('allCurve', all); drawBars();
  const legend=document.getElementById('legend'); legend.innerHTML=''; all.forEach((b,i)=>{ const e=document.createElement('span'); e.className='pill'; e.innerHTML=`<span class="dot" style="--c:${colors[i%colors.length]}"></span>${b}`; legend.appendChild(e); });
  document.querySelectorAll('canvas').forEach(attachHover);
  window.addEventListener('resize', () => { renderPair(); drawCurve('allCurve', all); drawBars(); });
}
init();
</script>
</body>
</html>
"""
    replacements = {
        "__ITERATIONS__": html.escape(iterations),
        "__SHARD_FILTER__": html.escape(shard_filter),
        "__SHARD_OPTIONS__": shard_options,
        "__SHARD_LABEL__": html.escape(shard_label),
        "__PARSED_SHARDS__": html.escape(parsed_shards_label),
        "__GENERATED_AT__": html.escape(generated_at),
        "__SCALE_OPTIONS__": scale_options,
        "__OK_COUNT__": str(len(ok)),
        "__MISSING_COUNT__": str(len(data["missing"])),
        "__BEST_BRANCH__": html.escape(ok_sorted[0]["branch"]) if ok_sorted else "n/a",
        "__BEST_BPB__": fmt(ok_sorted[0].get("roundtrip_val_bpb") or ok_sorted[0].get("final_val_bpb")) if ok_sorted else "n/a",
        "__MISSING_SECTION__": (
            '<section class="warn"><h2>Missing / incomplete</h2><ul>' + missing_html + "</ul></section>"
            if data["missing"]
            else ""
        ),
        "__TABLE_ROWS__": table_rows,
        "__PAYLOAD__": payload,
    }
    for key, value in replacements.items():
        document = document.replace(key, value)
    path.write_text(document, encoding="utf-8")


def main() -> None:
    iterations = sys.argv[1] if len(sys.argv) > 1 else "2000"
    shard_filter = normalize_shard_filter(sys.argv[2] if len(sys.argv) > 2 else "preferred")
    out_dir = Path("My_approch") / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect(iterations, shard_filter)
    stem = report_stem(iterations, shard_filter)
    data_path = out_dir / f"{stem}_data.json"
    csv_path = out_dir / f"{stem}_summary.csv"
    html_path = out_dir / f"{stem}.html"
    alias_html_path = out_dir / "stf_2k_report.html" if iterations == "2000" and shard_filter == "preferred" else None
    alias_csv_path = out_dir / "stf_2k_summary.csv" if iterations == "2000" and shard_filter == "preferred" else None
    index_path = out_dir / "index.html"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_csv(data, csv_path)
    write_html(data, html_path)
    if alias_html_path is not None:
        alias_html_path.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    if alias_csv_path is not None:
        alias_csv_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    index_scale_options = "".join(
        f'<option value="{scale}" {"selected" if scale == iterations else ""}>{label}</option>'
        for scale, label in [
            ("2000", "2k"),
            ("5000", "5k"),
            ("10000", "10k"),
            ("20000", "20k"),
        ]
    )
    index_shard_options = "".join(
        f'<option value="{value}" {"selected" if value == shard_filter else ""}>{label}</option>'
        for value, label in [
            ("preferred", "Prefer 80 shards / highest available"),
            ("80", "Exactly 80 shards"),
            ("1", "Exactly 1 shard"),
        ]
    )
    index_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>STF Analysis Index</title>
  <style>
    body {{ margin:0; min-height:100vh; background:radial-gradient(circle at 10% -10%, #1f4c58 0, transparent 28rem), linear-gradient(160deg,#05070b,#111827); color:#eef4ff; font-family:ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
    main {{ max-width:900px; margin:auto; padding:52px 22px; }}
    h1 {{ font-size:clamp(2.4rem,6vw,5rem); letter-spacing:-.06em; line-height:.92; margin:0 0 18px; }}
    p {{ color:#91a0b8; line-height:1.6; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:22px 0; }}
    a {{ display:block; border:1px solid #263448; border-radius:18px; background:#0b111a; color:#58d7e8; padding:16px; text-decoration:none; font-weight:800; }}
    a.primary {{ background:#58d7e8; color:#081018; border-color:#58d7e8; }}
    .controls {{ display:grid; grid-template-columns:160px 1fr auto; gap:10px; align-items:center; border:1px solid #263448; border-radius:20px; background:#0b111a; padding:14px; margin:22px 0; }}
    label {{ color:#91a0b8; font-weight:800; text-transform:uppercase; letter-spacing:.08em; font-size:.8rem; }}
    select, input {{ width:100%; border:1px solid #263448; border-radius:12px; background:#05070b; color:#eef4ff; padding:11px 12px; font-size:1rem; }}
    button {{ border:1px solid #58d7e8; border-radius:12px; background:#58d7e8; color:#081018; padding:11px 14px; font-size:1rem; font-weight:900; cursor:pointer; }}
    code {{ background:#0b111a; border:1px solid #263448; border-radius:6px; padding:1px 6px; color:#dce8ff; }}
    @media(max-width:760px) {{ .controls {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>STF Analysis</h1>
  <p>Latest generated report: <code>{html.escape(html_path.name)}</code></p>
  <div class="controls">
    <label for="iterationSelect">Iteration</label>
    <select id="iterationSelect">{index_scale_options}</select>
    <button id="loadIteration" type="button">Load / generate</button>
    <label for="shardSelect">Shards</label>
    <select id="shardSelect">{index_shard_options}</select>
    <span>Prefer 80 handles the 5k duplicate-run case.</span>
    <label for="customIterations">Custom</label>
    <input id="customIterations" inputmode="numeric" pattern="[0-9]*" placeholder="optional, e.g. 50000">
    <button id="loadCustomIteration" type="button">Use custom</button>
  </div>
  <p id="status">When opened through <code>python scripts/serve_stf_analysis.py</code>, this page can run the report generator for missing iteration data. In plain file mode it will show the command to run.</p>
  <div class="grid">
    <a class="primary" href="{html.escape(html_path.name)}">Open latest</a>
    <a href="stf_2000_branch_compare.html">2000 iterations</a>
    <a href="stf_5000_branch_compare.html">5000 iterations</a>
    <a href="stf_5000_shards80_branch_compare.html">5000 iterations, 80 shards</a>
    <a href="stf_5000_shards1_branch_compare.html">5000 iterations, 1 shard</a>
    <a href="stf_10000_branch_compare.html">10000 iterations</a>
    <a href="stf_20000_branch_compare.html">20000 iterations</a>
    <a href="stf_2k_report.html">2k alias</a>
  </div>
  <p>Generated: {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
  <p>Refresh command: <code>python scripts/generate_stf_analysis.py {html.escape(iterations)} {html.escape(shard_filter)}</code></p>
</main>
<script>
function selectedIteration() {{
  const custom = document.getElementById('customIterations').value.trim();
  return custom || document.getElementById('iterationSelect').value;
}}
function selectedShard() {{ return document.getElementById('shardSelect').value || 'preferred'; }}
function reportName(iter, shard) {{
  return shard === 'preferred' ? `stf_${{iter}}_branch_compare.html` : `stf_${{iter}}_shards${{shard}}_branch_compare.html`;
}}
function setStatus(html) {{ document.getElementById('status').innerHTML = html; }}
async function loadIteration() {{
  const iter = selectedIteration();
  const shard = selectedShard();
  if(!/^\\d+$/.test(iter)) {{ setStatus('Iteration must be numeric.'); return; }}
  const target = reportName(iter, shard);
  const command = `python scripts/generate_stf_analysis.py ${{iter}} ${{shard}}`;
  const serverMode = location.protocol === 'http:' || location.protocol === 'https:';
  if(!serverMode) {{
    setStatus(`Static file mode cannot execute Python. Run <code>${{command}}</code>, then open <a href="${{target}}">${{target}}</a>.`);
    return;
  }}
  setStatus(`Generating/loading <code>${{iter}}</code> with shard mode <code>${{shard}}</code> from branch logs...`);
  try {{
    const res = await fetch(`/api/report?iterations=${{encodeURIComponent(iter)}}&shards=${{encodeURIComponent(shard)}}`);
    const body = await res.json().catch(() => ({{}}));
    if(!res.ok || !body.ok) {{
      setStatus(`Generation failed. Run <code>${{command}}</code>. ${{body.error || ''}}`);
      return;
    }}
    window.location.href = target;
  }} catch (err) {{
    setStatus(`Could not reach the local analysis server. Run <code>python scripts/serve_stf_analysis.py</code> or run <code>${{command}}</code> manually.`);
  }}
}}
document.getElementById('loadIteration').onclick = loadIteration;
document.getElementById('loadCustomIteration').onclick = loadIteration;
</script>
</body>
</html>
""",
        encoding="utf-8",
    )

    ok = sorted(
        [record for record in data["records"] if record.get("status") == "ok"],
        key=lambda record: float(record.get("roundtrip_val_bpb") or record.get("final_val_bpb") or "inf"),
    )
    print(f"WROTE {html_path}")
    print(f"WROTE {csv_path}")
    print(f"WROTE {data_path}")
    if alias_html_path is not None:
        print(f"WROTE {alias_html_path}")
    print(f"WROTE {index_path}")
    for record in ok:
        print(
            f"{record['branch']}: final={fmt(record.get('final_val_bpb'))} "
            f"roundtrip={fmt(record.get('roundtrip_val_bpb'))} "
            f"shards={record.get('train_shards')} "
            f"artifact={fmt_bytes(record.get('total_submission_int8_zlib_bytes'))} "
            f"margin={fmt_bytes(record.get('artifact_margin_bytes'))} "
            f"best={fmt(record.get('best_val_bpb'))}@{record.get('best_step')} "
            f"step_avg={fmt(record.get('step_avg_ms'), 2)}ms "
            f"gpu={fmt(record.get('gpu_util_avg'), 1)}%"
        )
    for item in data["missing"]:
        print(f"MISSING {item['branch']}: {item['reason']}")


if __name__ == "__main__":
    main()

