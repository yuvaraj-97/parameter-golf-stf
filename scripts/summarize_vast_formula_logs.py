#!/usr/bin/env python3
"""Summarize partial Vast STF formula sweep logs and emit follow-up commands."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


KEY_VALUE_RE = re.compile(r"([A-Za-z_]+):([-+0-9.eE]+)")
STRING_VALUE_RE = re.compile(r"\b(compute_mode):([A-Za-z0-9_./-]+)")
VAL_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([-+0-9.eE]+)\s+val_bpb:([-+0-9.eE]+)")
TRAIN_RE = re.compile(r"step:(\d+)/(\d+)\s+train_loss:([-+0-9.eE]+)\s+train_time:(\d+)ms\s+step_avg:([-+0-9.eE]+)ms")
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip(?:_exact)?\s+val_loss:([-+0-9.eE]+)\s+val_bpb:([-+0-9.eE]+)")
STEP_RE = re.compile(r"step:(\d+)/(\d+)")
LAYER_SERIES_RE = re.compile(r"\b(active_by_layer|gate_by_layer|computed_by_layer|actual_skip_by_layer):([0-9:.,+-]+)")


@dataclass
class Variant:
    source: str
    branch: str
    score_fn: str
    run_id: str = ""
    final_loss: float | None = None
    final_bpb: float | None = None
    max_steps: int | None = None
    last_val_step: int | None = None
    last_val_bpb: float | None = None
    last_train_step: int | None = None
    last_train_loss: float | None = None
    last_step_avg_ms: float | None = None
    stats: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    layer_stats: dict[str, dict[int, float]] = field(default_factory=dict)
    telemetry: list[dict[str, object]] = field(default_factory=list)
    val_history: list[tuple[int, float]] = field(default_factory=list)
    train_history: list[tuple[int, float, float]] = field(default_factory=list)
    failed: bool = False

    @property
    def complete(self) -> bool:
        return self.final_bpb is not None and math.isfinite(self.final_bpb)

    def health_reason(self) -> str:
        active = self.stats.get("active_mean")
        gate = self.stats.get("gate_mean")
        score = self.stats.get("score_mean")
        if score is not None and not math.isfinite(score):
            return "bad_score"
        if active is not None and not (0.10 <= active <= 0.95):
            return f"active_mean={active:.4f}"
        if gate is not None and not (0.05 <= gate <= 0.98):
            return f"gate_mean={gate:.4f}"
        return "ok"

    def control_kind(self) -> str:
        if "gate_mean" in self.stats or "gate_by_layer" in self.layer_stats:
            return "gate"
        return "active"

    def active_proxy(self) -> float | None:
        if self.control_kind() == "gate":
            return self.stats.get("gate_mean")
        return self.stats.get("active_mean")

    def frozen_proxy(self) -> float | None:
        active = self.active_proxy()
        if active is None:
            return None
        return max(0.0, min(1.0, 1.0 - active))

    def final_layer_values(self) -> dict[int, float]:
        if self.control_kind() == "gate":
            return self.layer_stats.get("gate_by_layer", {})
        return self.layer_stats.get("active_by_layer", {})


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def parse_layer_series(line: str) -> dict[str, dict[int, float]]:
    parsed: dict[str, dict[int, float]] = {}
    for key, value in LAYER_SERIES_RE.findall(line):
        layers: dict[int, float] = {}
        for item in value.split(","):
            if ":" not in item:
                continue
            layer_raw, amount_raw = item.split(":", 1)
            try:
                layer = int(layer_raw)
                amount = float(amount_raw)
            except ValueError:
                continue
            if math.isfinite(amount):
                layers[layer] = amount
        if layers:
            parsed[key] = layers
    return parsed


def get_variant(variants: dict[tuple[str, str, str], Variant], source: str, branch: str, score_fn: str) -> Variant:
    key = (source, branch, score_fn)
    if key not in variants:
        variants[key] = Variant(source=source, branch=branch, score_fn=score_fn)
    return variants[key]


def parse_logs(paths: list[Path]) -> list[Variant]:
    variants: dict[tuple[str, str, str], Variant] = {}

    for path in paths:
        source = str(path)
        current_branch = ""
        current_score_fn = ""
        current_run_id = ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue

                if line.startswith("branch="):
                    current_branch = line.split("=", 1)[1].strip()
                    continue
                if line.startswith("score_fn="):
                    current_score_fn = line.split("=", 1)[1].strip()
                    if current_branch:
                        v = get_variant(variants, source, current_branch, current_score_fn)
                        v.run_id = current_run_id
                    continue
                if line.startswith("run_id="):
                    current_run_id = line.split("=", 1)[1].strip()
                    continue
                if line.startswith("- Branch:"):
                    current_branch = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("- STF score function:"):
                    current_score_fn = line.split(":", 1)[1].strip()
                    if current_branch:
                        get_variant(variants, source, current_branch, current_score_fn)
                    continue

                if not current_branch or not current_score_fn:
                    continue
                variant = get_variant(variants, source, current_branch, current_score_fn)
                if current_run_id:
                    variant.run_id = current_run_id

                val_match = VAL_RE.search(line)
                if val_match:
                    variant.last_val_step = int(val_match.group(1))
                    variant.max_steps = int(val_match.group(2))
                    variant.last_val_bpb = parse_float(val_match.group(4))
                    if variant.last_val_bpb is not None:
                        variant.val_history.append((variant.last_val_step, variant.last_val_bpb))

                train_match = TRAIN_RE.search(line)
                if train_match:
                    variant.last_train_step = int(train_match.group(1))
                    variant.max_steps = int(train_match.group(2))
                    variant.last_train_loss = parse_float(train_match.group(3))
                    variant.last_step_avg_ms = parse_float(train_match.group(5))
                    if variant.last_train_loss is not None and variant.last_step_avg_ms is not None:
                        variant.train_history.append(
                            (variant.last_train_step, variant.last_train_loss, variant.last_step_avg_ms)
                        )

                final_match = FINAL_RE.search(line)
                if final_match:
                    variant.final_loss = parse_float(final_match.group(1))
                    variant.final_bpb = parse_float(final_match.group(2))

                if "stf_stats" in line:
                    line_stats: dict[str, float] = {}
                    for key, value in KEY_VALUE_RE.findall(line):
                        parsed = parse_float(value)
                        if parsed is not None:
                            variant.stats[key] = parsed
                            line_stats[key] = parsed
                    for key, value in STRING_VALUE_RE.findall(line):
                        variant.labels[key] = value

                    line_layers = parse_layer_series(line)
                    for key, values in line_layers.items():
                        variant.layer_stats[key] = values

                    step_match = STEP_RE.search(line)
                    if step_match:
                        variant.max_steps = int(step_match.group(2))
                        variant.telemetry.append(
                            {
                                "step": int(step_match.group(1)),
                                "stats": line_stats,
                                "labels": dict(variant.labels),
                                "layers": line_layers,
                            }
                        )

                if "VARIANT FAILED" in line or "Traceback " in line:
                    variant.failed = True

    return list(variants.values())


def dedupe_variants(rows: list[Variant]) -> list[Variant]:
    deduped: dict[tuple[object, ...], Variant] = {}
    for variant in rows:
        key = (
            variant.branch,
            variant.score_fn,
            variant_steps(variant),
            round(variant.final_bpb or -1.0, 8),
            round(variant.last_val_bpb or -1.0, 8),
            round(variant.stats.get("active_mean", -1.0), 6),
            round(variant.stats.get("gate_mean", -1.0), 6),
            round(variant.stats.get("computed_token_ratio", -1.0), 6),
            round(variant.stats.get("actual_skip_ratio", -1.0), 6),
        )
        current = deduped.get(key)
        if current is None or ("/console.log" in current.source and "/console.log" not in variant.source):
            deduped[key] = variant
    return list(deduped.values())


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{100.0 * value:.{digits}f}%"


def print_table(title: str, rows: list[Variant]) -> None:
    print(title)
    print("branch | formula | final_bpb | last_step | last_bpb | active | gate | score | reason | source")
    print("-" * 111)
    for v in rows:
        print(
            " | ".join(
                [
                    v.branch,
                    v.score_fn,
                    fmt(v.final_bpb),
                    fmt(v.last_val_step),
                    fmt(v.last_val_bpb),
                    fmt(v.stats.get("active_mean")),
                    fmt(v.stats.get("gate_mean")),
                    fmt(v.stats.get("score_mean"), 6),
                    v.health_reason(),
                    v.source,
                ]
            )
        )
    print()


def print_baseline_comparison(rows: list[Variant], baseline: str) -> None:
    by_branch: dict[str, dict[str, Variant]] = defaultdict(dict)
    for variant in rows:
        by_branch[variant.branch][variant.score_fn] = variant

    print(f"Best completed formula compared with baseline '{baseline}'")
    print("branch | best_formula | best_bpb | baseline_bpb | delta_bpb | relative")
    print("-" * 78)
    for branch in sorted(by_branch):
        variants = [v for v in by_branch[branch].values() if v.final_bpb is not None]
        if not variants:
            continue
        base = by_branch[branch].get(baseline)
        best = min(variants, key=lambda v: v.final_bpb if v.final_bpb is not None else float("inf"))
        if base is None or base.final_bpb is None or best.final_bpb is None:
            print(f"{branch} | {best.score_fn} | {fmt(best.final_bpb)} | - | - | -")
            continue
        delta = base.final_bpb - best.final_bpb
        relative = 100.0 * delta / base.final_bpb if base.final_bpb else 0.0
        print(
            " | ".join(
                [
                    branch,
                    best.score_fn,
                    fmt(best.final_bpb),
                    fmt(base.final_bpb),
                    f"{delta:+.4f}",
                    f"{relative:+.2f}%",
                ]
            )
        )
    print()


def emit_commands(selected: list[Variant], iterations: int, val_every: int, telemetry: int) -> None:
    by_branch: dict[str, list[str]] = defaultdict(list)
    for variant in selected:
        if variant.score_fn not in by_branch[variant.branch]:
            by_branch[variant.branch].append(variant.score_fn)

    print("Suggested 2k commands")
    for branch, formulas in by_branch.items():
        formula_str = " ".join(formulas)
        print(
            f'STF_BRANCHES="{branch}" \\\n'
            f'STF_SCORE_FNS="{formula_str}" \\\n'
            f"ITERATIONS={iterations} \\\n"
            f"VAL_LOSS_EVERY={val_every} \\\n"
            "TRAIN_LOG_EVERY=100 \\\n"
            f"STF_TELEMETRY={telemetry} \\\n"
            "WARMUP_STEPS=0 \\\n"
            "MUON_BACKEND_STEPS=3 \\\n"
            "bash scripts/run_vast_formula_series.sh 2>&1 | tee -a vast_2k_filtered.log\n"
        )


def bpb_delta_class(variant: Variant, baseline_by_branch: dict[str, Variant]) -> str:
    base = baseline_by_branch.get(variant.branch)
    if base is None or base.final_bpb is None or variant.final_bpb is None:
        return "neutral"
    delta = base.final_bpb - variant.final_bpb
    if delta > 0.004:
        return "good"
    if delta < -0.004:
        return "bad"
    return "neutral"


def variant_slug(variant: Variant) -> str:
    raw = f"{variant.branch}-{variant.score_fn}"
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-")


def norm_good(value: float | None, low: float, high: float) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    if high <= low:
        return 1.0
    return max(0.0, min(1.0, (high - value) / (high - low)))


def ellipsize(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def render_outcome_dashboard(rows: list[Variant], width: int = 1600) -> str:
    rows = rows[:12]
    if not rows:
        return ""

    bpb_values = [v.final_bpb for v in rows if v.final_bpb is not None]
    loss_values = [v.last_train_loss for v in rows if v.last_train_loss is not None]
    speed_values = [v.last_step_avg_ms for v in rows if v.last_step_avg_ms is not None]
    if not bpb_values or not loss_values or not speed_values:
        return '<div class="empty-viz">Not enough train/validation/speed telemetry for the combined dashboard.</div>'

    bpb_low, bpb_high = min(bpb_values), max(bpb_values)
    loss_low, loss_high = min(loss_values), max(loss_values)
    speed_low, speed_high = min(speed_values), max(speed_values)
    row_h = 46
    top = 82
    left = 560
    bar_w = width - left - 48
    height = top + row_h * len(rows) + 38
    colors = {"quality": "#58f2d0", "learning": "#b8ff72", "speed": "#f3b45b"}
    parts = [
        f'<svg class="outcome-viz" viewBox="0 0 {width} {height}" role="img" aria-label="Quality learning speed comparison">',
        '<text x="24" y="34" class="outcome-title">One-glance outcome map</text>',
        '<text x="24" y="58" class="outcome-subtitle">Longer bars are better. Quality = lower validation BPB, learning = lower train loss, wall-clock = lower average iteration time.</text>',
        f'<text x="{left}" y="70" class="axis-label">quality</text>',
        f'<text x="{left + bar_w * 0.36}" y="70" class="axis-label">learning</text>',
        f'<text x="{left + bar_w * 0.70}" y="70" class="axis-label">wall-clock</text>',
    ]

    for idx, variant in enumerate(rows):
        y = top + idx * row_h
        quality = norm_good(variant.final_bpb, bpb_low, bpb_high)
        learning = norm_good(variant.last_train_loss, loss_low, loss_high)
        speed = norm_good(variant.last_step_avg_ms, speed_low, speed_high)
        label = ellipsize(f"{variant.branch} / {variant.score_fn}", 52)
        parts.extend(
            [
                f'<text x="24" y="{y + 18}" class="row-label">{html.escape(label)}</text>',
                f'<text x="24" y="{y + 36}" class="row-detail">BPB {fmt(variant.final_bpb)} | loss {fmt(variant.last_train_loss)} | {fmt(variant.last_step_avg_ms, 1)} ms</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w}" height="30" rx="15" class="outcome-track" />',
                f'<rect x="{left}" y="{y}" width="{bar_w * quality * 0.32:.1f}" height="30" rx="15" fill="{colors["quality"]}" opacity=".82" />',
                f'<rect x="{left + bar_w * 0.34}" y="{y}" width="{bar_w * learning * 0.30:.1f}" height="30" rx="15" fill="{colors["learning"]}" opacity=".78" />',
                f'<rect x="{left + bar_w * 0.66}" y="{y}" width="{bar_w * speed * 0.30:.1f}" height="30" rx="15" fill="{colors["speed"]}" opacity=".78" />',
            ]
        )

    parts.append("</svg>")
    return "".join(parts)


def render_speed_reality(rows: list[Variant]) -> str:
    by_branch: dict[str, list[float]] = defaultdict(list)
    for variant in rows:
        if variant.last_step_avg_ms is not None:
            by_branch[variant.branch].append(variant.last_step_avg_ms)
    if not by_branch:
        return ""

    branch_rows = []
    for branch, values in sorted(by_branch.items(), key=lambda item: sum(item[1]) / len(item[1])):
        avg_ms = sum(values) / len(values)
        branch_rows.append(
            f"<tr><td>{html.escape(branch)}</td><td>{fmt(avg_ms, 1)}ms</td><td>{fmt(min(values), 1)}ms</td><td>{fmt(max(values), 1)}ms</td></tr>"
        )

    compute_rows = []
    for variant in sorted(rows, key=lambda v: (v.stats.get("actual_skip_ratio", -1.0), -(v.frozen_proxy() or 0.0))):
        if "computed_token_ratio" not in variant.stats and "actual_skip_ratio" not in variant.stats:
            continue
        computed = variant.stats.get("computed_token_ratio")
        logical_frozen = variant.stats.get("frozen_token_ratio", variant.frozen_proxy())
        actual_skip = variant.stats.get("actual_skip_ratio")
        efficiency = variant.stats.get("skip_efficiency")
        compute_mode = variant.labels.get("compute_mode", "-")
        compute_rows.append(
            "<tr>"
            f"<td>{html.escape(variant.branch)}</td>"
            f"<td>{html.escape(variant.score_fn)}</td>"
            f"<td>{html.escape(compute_mode)}</td>"
            f"<td>{pct(logical_frozen)}</td>"
            f"<td>{pct(computed)}</td>"
            f"<td>{pct(actual_skip)}</td>"
            f"<td>{pct(efficiency)}</td>"
            "</tr>"
        )

    compute_table = ""
    if compute_rows:
        compute_table = f"""
      <div class="table-wrap compact">
        <table>
          <thead><tr><th>Branch</th><th>Formula</th><th>Compute Mode</th><th>Logically Frozen</th><th>Actually Computed</th><th>Actually Skipped</th><th>Skip Efficiency</th></tr></thead>
          <tbody>{''.join(compute_rows)}</tbody>
        </table>
      </div>
      <p class="note">Skip efficiency compares actual skipped compute against the theoretical skip implied by frozen tokens. Near 0% means the model is still paying for frozen tokens; near 100% means freezing is translating into compute savings.</p>
        """

    return f"""
    <div class="speed-card">
      <div>
        <h3>Does freezing make future layers faster?</h3>
        <p>Only if the implementation actually stops sending frozen tokens through later layer compute. If the code keeps full tensors and just masks, blends, or reuses frozen states, the model may be logically frozen but not much faster.</p>
        <p>Old logs show whole-step average time, not per-layer skip. New STF telemetry can now expose the key reality check directly: what percentage of token-layer work was logically frozen versus actually skipped.</p>
      </div>
      <div class="table-wrap compact">
        <table>
          <thead><tr><th>Branch</th><th>Avg Iter</th><th>Fastest</th><th>Slowest</th></tr></thead>
          <tbody>{''.join(branch_rows)}</tbody>
        </table>
      </div>
      {compute_table}
    </div>
    """


def variant_steps(variant: Variant) -> int | None:
    return variant.max_steps or variant.last_val_step or variant.last_train_step


def is_fixed_mlp_skip_075(variant: Variant) -> bool:
    if variant.branch != "codex/stf-mlp-skip-runner" or variant.score_fn != "relative_l2":
        return False
    budget = variant.stats.get("active_budget")
    if budget is not None:
        return 0.745 <= budget <= 0.755
    actual_skip = variant.stats.get("actual_skip_ratio")
    computed = variant.stats.get("computed_token_ratio")
    source_hint = "a075" in variant.source or "i5000_run79" in variant.source or "i10000_run76" in variant.source
    telemetry_hint = actual_skip is not None and 0.205 <= actual_skip <= 0.212
    computed_hint = computed is not None and 0.788 <= computed <= 0.795
    return source_hint or telemetry_hint or computed_hint


def is_adaptive_mlp_budget(variant: Variant) -> bool:
    return variant.branch == "codex/stf-adaptive-mlp-budget" and variant.score_fn == "relative_l2"


def best_by_steps(rows: list[Variant], predicate) -> dict[int, Variant]:
    grouped: dict[int, list[Variant]] = defaultdict(list)
    for variant in rows:
        steps = variant_steps(variant)
        if steps is not None and predicate(variant):
            grouped[steps].append(variant)
    best: dict[int, Variant] = {}
    for steps, variants in grouped.items():
        best[steps] = min(
            variants,
            key=lambda v: (
                v.final_bpb if v.final_bpb is not None else float("inf"),
                v.source,
            ),
        )
    return best


def render_validation_ladder(completed: list[Variant]) -> str:
    fixed = best_by_steps(completed, is_fixed_mlp_skip_075)
    adaptive = best_by_steps(completed, is_adaptive_mlp_budget)
    if not fixed and not adaptive:
        return ""

    rows: list[tuple[str, int, Variant]] = []
    for steps in sorted(fixed):
        if steps in {2000, 5000, 10000}:
            rows.append(("Fixed MLP skip, active 0.75", steps, fixed[steps]))
    for steps in sorted(adaptive):
        if steps in {500, 2000, 10000}:
            rows.append(("Adaptive MLP budget, 0.75 -> 0.70 -> 0.65", steps, adaptive[steps]))

    body = []
    for label, steps, variant in rows:
        reference = fixed.get(steps)
        quality_delta = None
        speed_delta = None
        if reference and reference is not variant:
            if reference.final_bpb is not None and variant.final_bpb is not None:
                quality_delta = variant.final_bpb - reference.final_bpb
            if reference.last_step_avg_ms is not None and variant.last_step_avg_ms is not None:
                speed_delta = (reference.last_step_avg_ms - variant.last_step_avg_ms) / reference.last_step_avg_ms

        if variant.branch == "codex/stf-adaptive-mlp-budget":
            verdict = "Best speed-quality tradeoff so far" if steps == 10000 else "Useful budget smoke"
        elif steps == 10000:
            verdict = "Best quality reference"
        else:
            verdict = "Reference run"

        body.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{steps:,}</td>"
            f"<td>{fmt(variant.final_bpb)}</td>"
            f"<td>{fmt(variant.last_val_bpb)}</td>"
            f"<td>{fmt(variant.last_step_avg_ms, 1)}ms</td>"
            f"<td>{pct(variant.stats.get('actual_skip_ratio'))}</td>"
            f"<td>{pct(variant.stats.get('computed_token_ratio'))}</td>"
            f"<td class=\"{('bad' if quality_delta and quality_delta > 0 else 'good') if quality_delta is not None else 'neutral'}\">{fmt(quality_delta)}</td>"
            f"<td class=\"{('good' if speed_delta and speed_delta > 0 else 'bad') if speed_delta is not None else 'neutral'}\">{pct(speed_delta)}</td>"
            f"<td>{html.escape(verdict)}</td>"
            "</tr>"
        )

    if not body:
        return ""

    return f"""
    <div class="ladder-card" id="validation-ladder">
      <div>
        <p class="eyebrow">Budget-aware validation ladder</p>
        <h2>What We Actually Proved</h2>
        <p class="note">500-step runs are treated as smoke tests. The 10k rows are the real quality signal. Quality delta is shown against the fixed active-0.75 MLP-skip run at the same iteration count when available.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Track</th><th>Iterations</th><th>Final BPB</th><th>Last Val BPB</th><th>Avg Iter</th><th>Actual Skip</th><th>Computed</th><th>BPB Cost</th><th>Speed Gain</th><th>Verdict</th></tr></thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
    </div>
    """


def clean_branch_name(raw: str) -> str | None:
    branch = raw.strip()
    if not branch or branch == "origin/HEAD":
        return None
    if branch.startswith("origin/"):
        branch = branch.removeprefix("origin/")
    return branch


def repo_branch_names() -> list[str]:
    try:
        result = subprocess.run(
            ["git", "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes/origin"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    branches = {name for line in result.stdout.splitlines() if (name := clean_branch_name(line))}
    return sorted(branches, key=branch_sort_key)


def repo_git_graph() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--graph",
                "--decorate",
                "--oneline",
                "--all",
                "--simplify-by-decoration",
                "--date-order",
                "--max-count=160",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "git graph unavailable"
    return result.stdout.strip() or "git graph unavailable"


def branch_sort_key(branch: str) -> tuple[int, str]:
    if branch == "main":
        return (0, branch)
    if branch == "baseline-repro":
        return (1, branch)
    if branch.startswith("codex/"):
        return (2, branch)
    if branch.endswith("-telemetry"):
        return (3, branch)
    if branch.startswith("stf-"):
        return (4, branch)
    return (5, branch)


def branch_status(branch: str) -> tuple[str, str]:
    if branch == "main":
        return ("main", "Current integration branch")
    if branch == "baseline-repro":
        return ("baseline", "OpenAI direct baseline context")
    if branch == "codex/stf-query-sparse-attn":
        return ("blocked", "Implementation branch, blocked in smoke")
    if branch in {"codex/stf-mlp-skip-runner", "codex/stf-adaptive-mlp-budget"}:
        return ("validated", "Validated true-skip branch")
    if branch.startswith("codex/"):
        return ("impl", "Implementation branch")
    if branch.endswith("-telemetry"):
        return ("telemetry", "Telemetry result branch")
    if branch == "stf-quantization":
        return ("archive", "Experiment branch with imported HTML reports")
    if branch.startswith("stf-"):
        return ("experiment", "STF experiment branch")
    return ("branch", "Repository branch")


def render_branch_inventory(completed: list[Variant]) -> str:
    branches = repo_branch_names()
    if not branches:
        branches = sorted({"main", "baseline-repro", *(v.branch for v in completed)}, key=branch_sort_key)
    completed_by_branch = defaultdict(int)
    for variant in completed:
        completed_by_branch[variant.branch] += 1

    cards = []
    for branch in branches:
        status, description = branch_status(branch)
        count = completed_by_branch.get(branch, 0)
        metric_text = f"{count} parsed run{'s' if count != 1 else ''}" if count else "no parsed run in this report"
        cards.append(
            f"""
            <div class="branch-chip">
              <span class="status {html.escape(status)}">{html.escape(status)}</span>
              <strong>{html.escape(branch)}</strong>
              <small>{html.escape(description)} | {html.escape(metric_text)}</small>
            </div>
            """
        )

    return f"""
      <div class="branch-inventory">
        {''.join(cards)}
      </div>
    """


def render_branch_compare_picker(completed: list[Variant]) -> str:
    by_branch: dict[str, list[Variant]] = defaultdict(list)
    for variant in completed:
        by_branch[variant.branch].append(variant)
    if not by_branch:
        return ""

    descriptions = {
        "codex/stf-mlp-skip-runner": "Validated true-skip MLP path. Gathers active token rows through the MLP and scatters them back so logical freezing becomes real MLP compute savings.",
        "codex/stf-adaptive-mlp-budget": "Validated adaptive active-budget schedule. Trades a small 10k BPB cost for higher actual skip than the fixed active-0.75 runner.",
        "codex/stf-query-sparse-attn": "Attempted sparse attention plus MLP true-skip path. Current smoke attempts are failed/blocked and need redesign before more pod time.",
        "stf-soft-freeze-telemetry": "Telemetry variant of soft-freeze. Shows high logical freeze while full-block-then-freeze still skips 0% actual compute.",
        "stf-recurrence-telemetry": "Telemetry recurrence variant showing compute reality for recurrent frozen-state paths.",
        "stf-learned-gate": "Learns a continuous gate instead of only using a fixed threshold. Good quality in some runs, but gate-open proxy can mean little true freezing.",
        "stf-learned-gate-telemetry": "Telemetry variant of learned-gate that separates logical gates from actual skipped work.",
    }

    records = []
    for branch, variants in sorted(
        by_branch.items(),
        key=lambda item: min(v.final_bpb if v.final_bpb is not None else float("inf") for v in item[1]),
    ):
        variants = sorted(
            variants,
            key=lambda v: (
                variant_steps(v) or 0,
                v.final_bpb if v.final_bpb is not None else float("inf"),
                v.score_fn,
            ),
        )
        best = min(variants, key=lambda v: v.final_bpb if v.final_bpb is not None else float("inf"))
        records.append(
            {
                "branch": branch,
                "description": descriptions.get(branch, branch_status(branch)[1]),
                "status": branch_status(branch)[0],
                "best_formula": best.score_fn,
                "best_final_bpb": best.final_bpb,
                "best_steps": variant_steps(best),
                "variants": [
                    {
                        "formula": variant.score_fn,
                        "steps": variant_steps(variant),
                        "final_bpb": variant.final_bpb,
                        "last_val_bpb": variant.last_val_bpb,
                        "step_ms": variant.last_step_avg_ms,
                        "actual_skip": variant.stats.get("actual_skip_ratio"),
                        "computed": variant.stats.get("computed_token_ratio"),
                        "frozen": variant.stats.get("frozen_token_ratio", variant.frozen_proxy()),
                        "active": variant.active_proxy(),
                        "source": variant.source,
                        "history": [{"step": step, "bpb": bpb} for step, bpb in variant.val_history],
                    }
                    for variant in variants
                ],
            }
        )
    payload = html.escape(json.dumps(records, separators=(",", ":")), quote=False)

    overview_cards = []
    for index, record in enumerate(records):
        histories = [
            variant["history"]
            for variant in record["variants"]
            if variant.get("history")
        ]
        points = [point for history in histories for point in history]
        steps = [point["step"] for point in points]
        bpbs = [point["bpb"] for point in points]
        lines = []
        if steps and bpbs:
            min_step, max_step = min(steps), max(steps)
            min_bpb, max_bpb = min(bpbs), max(bpbs)
            plot_w, plot_h = 280, 82
            for line_idx, history in enumerate(histories[:4]):
                coords = []
                for point in history:
                    x = 20 + (0 if max_step == min_step else (point["step"] - min_step) / (max_step - min_step) * plot_w)
                    y = 14 + (0 if max_bpb == min_bpb else (max_bpb - point["bpb"]) / (max_bpb - min_bpb) * plot_h)
                    coords.append(f"{x:.1f},{y:.1f}")
                color = ["#58f2d0", "#b8ff72", "#f3b45b", "#4aa8ff"][line_idx % 4]
                lines.append(f'<polyline points="{" ".join(coords)}" stroke="{color}" />')
        mini_lines = "".join(lines) if lines else '<text x="160" y="62">No curve</text>'
        mini_svg = (
            '<svg class="branch-mini" viewBox="0 0 320 120" role="img" aria-label="Branch validation curve">'
            '<rect x="20" y="14" width="280" height="82" rx="12" />'
            '<line x1="20" y1="96" x2="300" y2="96" />'
            '<line x1="20" y1="14" x2="20" y2="96" />'
            f"{mini_lines}"
            "</svg>"
        )
        best_final = fmt(record.get("best_final_bpb") if isinstance(record.get("best_final_bpb"), float) else None)
        actual_skip_values = [
            variant.get("actual_skip")
            for variant in record["variants"]
            if isinstance(variant.get("actual_skip"), (int, float))
        ]
        max_actual_skip = max(actual_skip_values) if actual_skip_values else None
        overview_cards.append(
            f"""
            <article class="branch-overview-card">
              <div class="branch-overview-head">
                <span class="status {html.escape(str(record["status"]))}">{html.escape(str(record["status"]))}</span>
                <strong>{html.escape(record["branch"])}</strong>
              </div>
              {mini_svg}
              <p>{html.escape(record["description"])}</p>
              <div class="branch-stats">
                <span class="branch-stat"><strong>{best_final}</strong><span>best final BPB</span></span>
                <span class="branch-stat"><strong>{fmt(record.get("best_steps") if isinstance(record.get("best_steps"), int) else None)}</strong><span>best run steps</span></span>
                <span class="branch-stat"><strong>{len(record["variants"])}</strong><span>completed variants</span></span>
                <span class="branch-stat"><strong>{pct(max_actual_skip)}</strong><span>max actual skip</span></span>
              </div>
            </article>
            """
        )

    options = "".join(
        f'<option value="{html.escape(record["branch"])}">{html.escape(record["branch"])}</option>'
        for record in records
    )
    default_a = html.escape(records[0]["branch"])
    default_b = html.escape(records[1]["branch"] if len(records) > 1 else records[0]["branch"])

    return f"""
    <div class="compare-card" id="branch-explorer">
      <div class="compare-copy">
        <p class="eyebrow">Branch graph and two-way comparison</p>
        <h2>Branch Explorer</h2>
        <p class="note">Every branch gets a mini validation graph below. Pick any two branches in the selectors to compare their curves, stats, description, and run table directly.</p>
      </div>
      <div class="branch-overview-grid">{''.join(overview_cards)}</div>
      <div class="compare-controls two">
        <label for="branchASelect">Branch A</label>
        <select id="branchASelect">{options}</select>
        <label for="branchBSelect">Branch B</label>
        <select id="branchBSelect">{options}</select>
      </div>
      <div class="branch-chart-wrap">
        <svg id="branchCompareChart" class="branch-chart" viewBox="0 0 1120 420" role="img" aria-label="Selected branch validation BPB curves"></svg>
      </div>
      <div class="branch-panels">
        <article class="branch-detail" id="branchADetail"></article>
        <article class="branch-detail" id="branchBDetail"></article>
      </div>
      <article class="branch-detail branch-delta" id="branchDeltaDetail"></article>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Branch</th><th>Formula</th><th>Iterations</th><th>Final BPB</th><th>Last BPB</th><th>Avg Iter</th><th>Actual Skip</th><th>Computed</th><th>Source</th></tr></thead>
          <tbody id="branchCompareRows"></tbody>
        </table>
      </div>
      <script type="application/json" id="branchCompareData">{payload}</script>
      <script type="application/json" id="branchCompareDefaults">{{"a":"{default_a}","b":"{default_b}"}}</script>
    </div>
    """


def render_branch_tree(completed: list[Variant]) -> str:
    soft_done = sorted({v.score_fn for v in completed if v.branch == "stf-soft-freeze-telemetry"})
    soft_done_text = ", ".join(soft_done) if soft_done else "waiting for completed telemetry logs"
    fixed_done = sorted(variant_steps(v) for v in completed if is_fixed_mlp_skip_075(v) and variant_steps(v) is not None)
    adaptive_done = sorted(variant_steps(v) for v in completed if is_adaptive_mlp_budget(v) and variant_steps(v) is not None)
    fixed_text = ", ".join(f"{steps // 1000}k" if steps >= 1000 else str(steps) for steps in sorted(set(fixed_done))) or "waiting"
    adaptive_text = ", ".join(f"{steps // 1000}k" if steps >= 1000 else str(steps) for steps in sorted(set(adaptive_done))) or "waiting"
    return f"""
    <section class="branch-tree-card" id="branch-tree">
      <div>
        <p class="eyebrow">Implementation map</p>
        <h2>Git Branch Tree</h2>
        <p class="note">Experiment-result branches stay separate from implementation branches. `baseline-repro` is included as the OpenAI direct baseline context, and the full repo branch inventory below is generated from local and remote git refs.</p>
      </div>
      <pre class="branch-tree">main
├── baseline-repro <span class="status baseline">baseline</span>
│   └── context: OpenAI direct baseline / control branch
│
├── stf-soft-freeze-telemetry <span class="status done">done</span>
│   ├── done: 2k {html.escape(soft_done_text)}
│   ├── done: full_block_then_freeze telemetry
│   └── result: logical freeze can be high while actual compute skipped is 0%
│
├── codex/stf-true-skip-foundation <span class="status done">done</span>
│   ├── done: pre-block active-mask foundation
│   ├── done: predictor / compute telemetry plumbing
│   └── result: base for true-skip implementations
│
├── codex/stf-mlp-skip-runner <span class="status done">validated</span>
│   ├── done: MLP active-row gather/scatter
│   ├── done: relative_l2 active 0.75 at {html.escape(fixed_text)}
│   └── result: best quality reference, actual_skip_ratio about 20.8%
│
├── codex/stf-adaptive-mlp-budget <span class="status done">validated</span>
│   ├── done: active budget schedule 0.75 -> 0.70 -> 0.65
│   ├── done: relative_l2 at {html.escape(adaptive_text)}
│   └── result: speed-quality tradeoff, actual_skip_ratio about 29.2%
│
├── codex/stf-query-sparse-attn <span class="status blocked">blocked</span>
│   ├── blocked: backend/OOM/stall before reliable 500-step smoke
│   └── next: redesign locally before spending more pod time
│
└── codex/stf-kv-reuse-experiment <span class="status todo">todo</span>
    ├── todo: cache/reuse frozen K/V or states
    └── todo: only run if previous tracks succeed</pre>
      <h3>Actual Git Ref Graph</h3>
      <p class="note">This graph is generated from git refs, so it shows branch split ancestry rather than just names. It intentionally includes local and remote decorations.</p>
      <pre class="git-graph">{html.escape(repo_git_graph())}</pre>
      <h3>All Repo Branches</h3>
      <p class="note">This list includes local branches and `origin/*` branches after stripping the remote prefix, so the report keeps every branch in context.</p>
      {render_branch_inventory(completed)}
    </section>
    """


def discover_run_log_paths() -> list[Path]:
    paths: set[Path] = set()
    paths.update(Path(".").glob("vast*.log"))
    vast_root = Path("vast") / "experiments"
    if vast_root.exists():
        paths.update(vast_root.rglob("console.log"))
        paths.update(vast_root.rglob("train.log"))
    return sorted(paths, key=lambda path: str(path))


def run_group_key(path: Path) -> str:
    if path.name in {"console.log", "train.log"} and len(path.parts) > 1:
        return str(path.parent)
    return str(path)


def branch_from_path(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 4 and parts[0] == "vast" and parts[1] == "experiments":
        if len(parts) >= 6 and parts[3] == "codex":
            return "codex/" + parts[4]
        return parts[3]
    stem = path.stem
    if stem.startswith("vast_"):
        return stem.removeprefix("vast_")
    return path.parent.name or stem


def parse_run_log_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(path),
        "group": run_group_key(path),
        "branch": branch_from_path(path),
        "score_fn": "-",
        "last_val_step": None,
        "max_steps": None,
        "best_val_bpb": None,
        "final_bpb": None,
        "last_step_avg_ms": None,
        "status": "seen",
    }
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if line.startswith("branch=") or line.startswith("- Branch:"):
                    summary["branch"] = line.split("=", 1)[-1].split(":", 1)[-1].strip()
                if line.startswith("score_fn=") or line.startswith("- STF score function:"):
                    summary["score_fn"] = line.split("=", 1)[-1].split(":", 1)[-1].strip()
                val_match = VAL_RE.search(line)
                if val_match:
                    step = int(val_match.group(1))
                    val_bpb = parse_float(val_match.group(4))
                    summary["last_val_step"] = step
                    summary["max_steps"] = int(val_match.group(2))
                    current_best = summary.get("best_val_bpb")
                    if val_bpb is not None and (current_best is None or val_bpb < current_best):
                        summary["best_val_bpb"] = val_bpb
                train_match = TRAIN_RE.search(line)
                if train_match:
                    summary["max_steps"] = int(train_match.group(2))
                    summary["last_step_avg_ms"] = parse_float(train_match.group(5))
                final_match = FINAL_RE.search(line)
                if final_match:
                    summary["final_bpb"] = parse_float(final_match.group(2))
                    summary["status"] = "completed"
                lowered = line.lower()
                if "variant failed" in lowered or "traceback " in lowered or "out of memory" in lowered:
                    summary["status"] = "failed"
    except OSError:
        summary["status"] = "unreadable"
    if summary["status"] == "seen" and summary.get("best_val_bpb") is not None:
        summary["status"] = "partial"
    return summary


def render_all_run_inventory() -> str:
    paths = discover_run_log_paths()
    if not paths:
        return ""
    parsed_variants = parse_logs(paths)
    parsed_sources = {variant.source for variant in parsed_variants}
    parsed_groups = {run_group_key(Path(source)) for source in parsed_sources}
    rows_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for variant in parsed_variants:
        source_path = Path(variant.source)
        best_val_bpb = min((bpb for _, bpb in variant.val_history), default=None)
        row_key = (
            variant.branch,
            variant.score_fn,
            variant.last_val_step,
            variant_steps(variant),
            round(best_val_bpb, 6) if best_val_bpb is not None else None,
            round(variant.final_bpb, 6) if variant.final_bpb is not None else None,
        )
        existing = rows_by_key.get(row_key)
        if existing is None:
            rows_by_key[row_key] = {
                "group": run_group_key(source_path),
                "branch": variant.branch,
                "score_fn": variant.score_fn,
                "last_val_step": variant.last_val_step,
                "max_steps": variant_steps(variant),
                "best_val_bpb": best_val_bpb,
                "final_bpb": variant.final_bpb,
                "last_step_avg_ms": variant.last_step_avg_ms,
                "status": "failed" if variant.failed else "completed" if variant.complete else "partial" if variant.last_val_bpb is not None else "seen",
                "files": [source_path.name],
                "groups": [run_group_key(source_path)],
            }
        else:
            existing.setdefault("files", []).append(source_path.name)
            existing.setdefault("groups", []).append(run_group_key(source_path))
    generic_paths = [path for path in paths if str(path) not in parsed_sources and run_group_key(path) not in parsed_groups]
    for path in generic_paths:
        summary = parse_run_log_summary(path)
        row_key = (
            summary.get("branch"),
            summary.get("score_fn"),
            summary.get("last_val_step"),
            summary.get("max_steps"),
            round(float(summary["best_val_bpb"]), 6) if isinstance(summary.get("best_val_bpb"), float) else None,
            round(float(summary["final_bpb"]), 6) if isinstance(summary.get("final_bpb"), float) else None,
            summary.get("status"),
        )
        existing = rows_by_key.get(row_key)
        if existing is None:
            summary["files"] = [path.name]
            summary["groups"] = [run_group_key(path)]
            rows_by_key[row_key] = summary
        else:
            existing.setdefault("files", []).append(path.name)
            existing.setdefault("groups", []).append(run_group_key(path))
    rows = list(rows_by_key.values())
    rows.sort(key=lambda item: (str(item.get("branch")), str(item.get("group")), str(item.get("score_fn"))))
    completed = sum(1 for row in rows if row.get("status") == "completed")
    partial = sum(1 for row in rows if row.get("status") == "partial")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    branches = len({str(row.get("branch")) for row in rows})

    cards = []
    for row in rows:
        status = str(row.get("status") or "seen")
        status_class = "done" if status == "completed" else "blocked" if status == "failed" else "running" if status == "partial" else "archive"
        last_step = f"{fmt(row.get('last_val_step') if isinstance(row.get('last_val_step'), int) else None)} / {fmt(row.get('max_steps') if isinstance(row.get('max_steps'), int) else None)}"
        files = row.get("files") if isinstance(row.get("files"), list) else [row.get("files")]
        groups = row.get("groups") if isinstance(row.get("groups"), list) else [row.get("group") or row.get("path")]
        files_label = ", ".join(html.escape(str(item)) for item in files[:3] if item)
        if len(files) > 3:
            files_label += f" +{len(files) - 3} more"
        groups_label = " | ".join(html.escape(str(item)) for item in groups[:2] if item)
        if len(groups) > 2:
            groups_label += f" | +{len(groups) - 2} more"
        cards.append(
            "<article class=\"run-row\">"
            "<div class=\"run-main\">"
            f"<strong>{html.escape(str(row.get('branch') or '-'))}</strong>"
            f"<span>{html.escape(str(row.get('score_fn') or '-'))}</span>"
            f"<span class=\"status {status_class}\">{html.escape(status)}</span>"
            "</div>"
            "<div class=\"run-metrics\">"
            f"<span><b>{html.escape(last_step)}</b><small>steps</small></span>"
            f"<span><b>{fmt(row.get('best_val_bpb') if isinstance(row.get('best_val_bpb'), float) else None)}</b><small>best BPB</small></span>"
            f"<span><b>{fmt(row.get('final_bpb') if isinstance(row.get('final_bpb'), float) else None)}</b><small>final BPB</small></span>"
            f"<span><b>{fmt(row.get('last_step_avg_ms') if isinstance(row.get('last_step_avg_ms'), float) else None, 1)}ms</b><small>avg iter</small></span>"
            "</div>"
            "<div class=\"run-paths\">"
            f"<span>{files_label or '-'}</span>"
            f"<code>{groups_label or '-'}</code>"
            "</div>"
            "</article>"
        )
    return f"""
    <section class="compare-card" id="all-run-inventory">
      <p class="eyebrow">Across-branch artifact audit</p>
      <h2>All Discovered Run Logs</h2>
      <p class="note">This ledger includes owned STF/Vast artifacts: top-level Vast logs and archived Vast experiment console/train logs. The separate non-STF records archive is intentionally excluded.</p>
      <div class="metrics">
        <div class="metric"><strong>{len(rows)}</strong><span>run groups discovered</span></div>
        <div class="metric"><strong>{branches}</strong><span>branches or run families</span></div>
        <div class="metric"><strong>{completed}</strong><span>completed artifacts</span></div>
        <div class="metric warn"><strong>{partial + failed}</strong><span>partial or failed artifacts</span></div>
      </div>
      <details>
        <summary>Show all discovered runs</summary>
        <div class="run-ledger">{''.join(cards)}</div>
      </details>
    </section>
    """


def render_layer_svg(variant: Variant, width: int = 720, height: int = 260) -> str:
    values = variant.final_layer_values()
    if not values:
        return '<div class="empty-viz">No per-layer telemetry found.</div>'

    layers = sorted(values)
    panel_w = 58
    gap = 36
    top = 36
    bottom = height - 42
    left = max(52, (width - (len(layers) * panel_w + (len(layers) - 1) * gap)) // 2)
    token_count = 12
    lines: list[str] = []

    for idx in range(len(layers) - 1):
        x1 = left + idx * (panel_w + gap) + panel_w
        x2 = left + (idx + 1) * (panel_w + gap)
        for token in range(token_count):
            y1 = top + token * ((bottom - top) / (token_count - 1))
            drift = ((token * 7 + idx * 3) % token_count) - token_count // 2
            y2 = top + min(token_count - 1, max(0, token + drift // 3)) * ((bottom - top) / (token_count - 1))
            opacity = 0.12 + 0.10 * ((token + idx) % 3)
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'class="neural-line" opacity="{opacity:.2f}" />'
            )

    panels: list[str] = []
    for idx, layer in enumerate(layers):
        active = max(0.0, min(1.0, values[layer]))
        frozen = 1.0 - active
        x = left + idx * (panel_w + gap)
        active_h = (bottom - top) * active
        frozen_h = (bottom - top) * frozen
        y_active = bottom - active_h
        panels.append(
            "\n".join(
                [
                    f'<g class="layer-node" transform="translate({x:.1f},0)">',
                    f'<rect x="0" y="{top}" width="{panel_w}" height="{bottom - top:.1f}" rx="14" class="layer-shell" />',
                    f'<rect x="0" y="{top}" width="{panel_w}" height="{frozen_h:.1f}" rx="14" class="frozen-fill" />',
                    f'<rect x="0" y="{y_active:.1f}" width="{panel_w}" height="{active_h:.1f}" rx="14" class="active-fill" />',
                    f'<text x="{panel_w / 2:.1f}" y="{top - 12}" class="layer-label">L{layer}</text>',
                    f'<text x="{panel_w / 2:.1f}" y="{bottom + 24}" class="layer-pct">{pct(frozen, 0)} frozen</text>',
                    "</g>",
                ]
            )
        )

    return (
        f'<svg class="layer-viz" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Layer freeze diagram for {html.escape(variant.branch)} {html.escape(variant.score_fn)}">'
        '<defs><filter id="softGlow"><feGaussianBlur stdDeviation="2.4" result="blur"/>'
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
        + "".join(lines)
        + "".join(panels)
        + "</svg>"
    )


def render_telemetry_sparkline(variant: Variant, width: int = 220, height: int = 54) -> str:
    points: list[tuple[int, float]] = []
    for item in variant.telemetry:
        stats = item.get("stats", {})
        if not isinstance(stats, dict):
            continue
        value = stats.get("gate_mean" if variant.control_kind() == "gate" else "active_mean")
        step = item.get("step")
        if isinstance(step, int) and isinstance(value, float):
            points.append((step, max(0.0, min(1.0, value))))
    if len(points) < 2:
        return ""

    min_step = min(step for step, _ in points)
    max_step = max(step for step, _ in points)
    span = max(1, max_step - min_step)
    coords = []
    for step, value in points:
        x = 8 + (width - 16) * (step - min_step) / span
        y = 8 + (height - 16) * (1.0 - value)
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" aria-label="active trend">'
        f'<polyline points="{" ".join(coords)}" />'
        '<line x1="8" y1="8" x2="212" y2="8" class="spark-guide" />'
        '<line x1="8" y1="46" x2="212" y2="46" class="spark-guide" />'
        "</svg>"
    )


def sampled_telemetry_frames(variant: Variant, max_frames: int = 6) -> list[tuple[int, dict[int, float]]]:
    frames: list[tuple[int, dict[int, float]]] = []
    layer_key = "gate_by_layer" if variant.control_kind() == "gate" else "active_by_layer"
    mean_key = "gate_mean" if variant.control_kind() == "gate" else "active_mean"
    fallback_layers = sorted(variant.final_layer_values()) or [3, 4, 5, 6, 7, 8]
    for item in variant.telemetry:
        step = item.get("step")
        layers = item.get("layers", {})
        stats = item.get("stats", {})
        if not isinstance(step, int) or not isinstance(layers, dict) or not isinstance(stats, dict):
            continue
        layer_values = layers.get(layer_key)
        if isinstance(layer_values, dict) and layer_values:
            values = {int(layer): float(value) for layer, value in layer_values.items()}
        else:
            mean = stats.get(mean_key)
            if not isinstance(mean, float):
                continue
            values = {layer: mean for layer in fallback_layers}
        frames.append((step, values))

    if len(frames) <= max_frames:
        return frames
    indexes = sorted({round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames)})
    return [frames[i] for i in indexes]


def render_freeze_animation(variant: Variant, width: int = 460, height: int = 250) -> str:
    frames = sampled_telemetry_frames(variant)
    if len(frames) < 2:
        return '<div class="empty-viz">Not enough telemetry frames to animate.</div>'

    layers = sorted({layer for _, values in frames for layer in values}) or [3, 4, 5, 6, 7, 8]
    token_count = 12
    top = 46
    bottom = height - 42
    left = 58
    right = width - 58
    x_gap = (right - left) / max(1, len(layers) - 1)
    y_gap = (bottom - top) / max(1, token_count - 1)
    key_times = ";".join(f"{i / (len(frames) - 1):.3f}" for i in range(len(frames)))
    step_values = " | ".join(str(step) for step, _ in frames)
    parts = [
        f'<svg class="freeze-movie" viewBox="0 0 {width} {height}" role="img" aria-label="Token freeze animation for {html.escape(variant.branch)} {html.escape(variant.score_fn)}">',
        f'<text x="20" y="24" class="movie-title">{html.escape(variant.score_fn)}</text>',
        f'<text x="{width - 20}" y="24" class="movie-step">steps {html.escape(step_values)}</text>',
    ]

    # Draw a lightweight network first; token nodes are drawn on top.
    for layer_idx in range(len(layers) - 1):
        x1 = left + layer_idx * x_gap
        x2 = left + (layer_idx + 1) * x_gap
        for token_idx in range(token_count):
            y1 = top + token_idx * y_gap
            y2_idx = min(token_count - 1, max(0, token_idx + ((token_idx + layer_idx) % 3) - 1))
            y2 = top + y2_idx * y_gap
            opacities = []
            for _, values in frames:
                a1 = max(0.0, min(1.0, values.get(layers[layer_idx], 1.0)))
                a2 = max(0.0, min(1.0, values.get(layers[layer_idx + 1], 1.0)))
                active_1 = token_idx < round(a1 * token_count)
                active_2 = y2_idx < round(a2 * token_count)
                opacities.append("0.42" if active_1 and active_2 else "0.09")
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="movie-edge">'
                f'<animate attributeName="opacity" values="{";".join(opacities)}" keyTimes="{key_times}" dur="4.8s" repeatCount="indefinite" />'
                "</line>"
            )

    for layer_idx, layer in enumerate(layers):
        x = left + layer_idx * x_gap
        parts.append(f'<text x="{x:.1f}" y="{height - 14}" class="movie-layer">L{layer}</text>')
        active_values = [max(0.0, min(1.0, values.get(layer, 1.0))) for _, values in frames]
        frozen_labels = [f"{100.0 * (1.0 - active):.0f}%" for active in active_values]
        parts.extend(
            [
                f'<text x="{x:.1f}" y="40" class="movie-freeze-pct">{frozen_labels[0]} frozen'
                f'<animate attributeName="opacity" values="1;.65;1" dur="4.8s" repeatCount="indefinite" />'
                "</text>"
            ]
        )
        for token_idx in range(token_count):
            y = top + token_idx * y_gap
            fill_values = []
            radius_values = []
            opacity_values = []
            for active in active_values:
                is_active = token_idx < round(active * token_count)
                fill_values.append("#58f26f" if is_active else "#4aa8ff")
                radius_values.append("5.2" if is_active else "4.2")
                opacity_values.append("1" if is_active else "0.72")
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius_values[0]}" class="movie-node">'
                f'<animate attributeName="fill" values="{";".join(fill_values)}" keyTimes="{key_times}" dur="4.8s" repeatCount="indefinite" />'
                f'<animate attributeName="r" values="{";".join(radius_values)}" keyTimes="{key_times}" dur="4.8s" repeatCount="indefinite" />'
                f'<animate attributeName="opacity" values="{";".join(opacity_values)}" keyTimes="{key_times}" dur="4.8s" repeatCount="indefinite" />'
                "</circle>"
            )

    parts.append("</svg>")
    return "".join(parts)


def write_html_report(rows: list[Variant], selected: list[Variant], output: Path, baseline: str) -> None:
    completed = [v for v in rows if v.complete and not v.failed]
    baseline_by_branch = {v.branch: v for v in completed if v.score_fn == baseline}
    branch_count = len({v.branch for v in completed})
    formula_count = len({v.score_fn for v in completed})
    active_values = [v.active_proxy() for v in completed if v.active_proxy() is not None]
    frozen_values = [1.0 - v for v in active_values]
    avg_frozen = sum(frozen_values) / len(frozen_values) if frozen_values else None
    hard_freeze = [v for v in completed if v.control_kind() == "active"]
    hard_avg = None
    if hard_freeze:
        hard_vals = [v.frozen_proxy() for v in hard_freeze if v.frozen_proxy() is not None]
        hard_avg = sum(hard_vals) / len(hard_vals) if hard_vals else None
    logical_frozen_values = [
        v.stats.get("frozen_token_ratio", v.frozen_proxy())
        for v in completed
        if v.stats.get("frozen_token_ratio", v.frozen_proxy()) is not None
    ]
    actual_skip_values = [v.stats["actual_skip_ratio"] for v in completed if "actual_skip_ratio" in v.stats]
    max_logical_frozen = max(logical_frozen_values) if logical_frozen_values else hard_avg
    max_actual_skip = max(actual_skip_values) if actual_skip_values else None
    has_actual_skip = max_actual_skip is not None and max_actual_skip > 0.0
    hero_title = (
        "MLP true-skip is validated; adaptive budget is the speed tradeoff."
        if has_actual_skip
        else "Tokens freeze logically, but compute is not reduced yet."
    )
    hero_body = (
        "The fixed <code>mlp_active_rows</code> run gives the best quality reference, while the adaptive "
        "0.75 -> 0.70 -> 0.65 budget raises actual skip to about 29% with a small 10k BPB cost. "
        "Attention remains dense, so this report separates logical freeze, real skipped compute, and wall-clock speed."
        if has_actual_skip
        else "The completed telemetry runs show real logical freezing, but the current implementation still "
        "runs the full transformer block first and freezes/blends afterward. Treat <b>logical freeze</b> as "
        "a model-state signal, not as speed saved, until <code>actual_skip_ratio</code> rises above zero."
    )

    best = completed[0] if completed else None
    ten_k_completed = sum(1 for variant in completed if variant_steps(variant) == 10000)
    visual_rows = selected or completed[:12]
    rows_html = []
    for rank, variant in enumerate(completed, start=1):
        base = baseline_by_branch.get(variant.branch)
        delta = None
        if base and base.final_bpb is not None and variant.final_bpb is not None:
            delta = base.final_bpb - variant.final_bpb
        delta_text = f"{delta:+.4f}" if delta is not None else "-"
        compute_mode = variant.labels.get("compute_mode", "-")
        computed_ratio = variant.stats.get("computed_token_ratio")
        actual_skip = variant.stats.get("actual_skip_ratio")
        skip_efficiency = variant.stats.get("skip_efficiency")
        rows_html.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{html.escape(variant.branch)}</td>"
            f"<td>{html.escape(variant.score_fn)}</td>"
            f"<td>{fmt(variant.final_bpb)}</td>"
            f"<td class=\"{bpb_delta_class(variant, baseline_by_branch)}\">{delta_text}</td>"
            f"<td>{fmt(variant.last_train_loss)}</td>"
            f"<td>{fmt(variant.last_step_avg_ms, 1)}ms</td>"
            f"<td>{pct(variant.frozen_proxy())}</td>"
            f"<td>{pct(variant.active_proxy())}</td>"
            f"<td>{html.escape(compute_mode)}</td>"
            f"<td>{pct(computed_ratio)}</td>"
            f"<td>{pct(actual_skip)}</td>"
            f"<td>{pct(skip_efficiency)}</td>"
            f"<td>{html.escape(variant.control_kind())}</td>"
            f"<td>{html.escape(variant.health_reason())}</td>"
            "</tr>"
        )

    outcome_dashboard = render_outcome_dashboard(completed)
    speed_reality = render_speed_reality(completed)
    branch_tree = render_branch_tree(completed)
    validation_ladder = render_validation_ladder(completed)
    branch_compare = render_branch_compare_picker(completed)
    all_run_inventory = render_all_run_inventory()
    movie_cards = []
    for variant in visual_rows[:8]:
        movie_cards.append(
            f"""
            <article class="movie-card">
              <div>
                <p class="eyebrow">{html.escape(variant.branch)}</p>
                <h3>{pct(variant.frozen_proxy())} final frozen</h3>
                <p class="small">A looping replay of the logged telemetry. Every token starts green/active; as training advances, frozen tokens cool into blue and their network edges dim.</p>
              </div>
              {render_freeze_animation(variant)}
            </article>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>STF Token Freezing Report</title>
  <style>
    :root {{
      --bg: #050806;
      --panel: rgba(14, 22, 18, 0.86);
      --panel-2: rgba(22, 30, 25, 0.72);
      --text: #eef8ed;
      --muted: #9fb2a5;
      --line: rgba(188, 255, 214, 0.15);
      --active: #58f26f;
      --frozen: #4aa8ff;
      --good: #7dff9f;
      --bad: #ff817d;
    }}
    * {{ box-sizing: border-box; letter-spacing: 0 !important; }}
    html {{ scroll-padding-bottom: 110px; }}
    body {{
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 12% 0%, rgba(88, 242, 208, 0.20), transparent 30rem),
        radial-gradient(circle at 88% 12%, rgba(243, 180, 91, 0.14), transparent 26rem),
        linear-gradient(135deg, #020302, var(--bg) 48%, #0d100a);
      font-family: Avenir Next, Avenir, Optima, Candara, sans-serif;
    }}
    main {{ width: min(1760px, calc(100vw - 32px)); margin: 0 auto; padding: 46px 0 150px; }}
    .hero {{
      border: 1px solid var(--line);
      border-radius: 32px;
      padding: 34px;
      background: linear-gradient(135deg, rgba(11, 18, 14, 0.94), rgba(20, 29, 23, 0.76));
      box-shadow: 0 26px 90px rgba(0, 0, 0, 0.38);
      overflow: hidden;
      position: relative;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -10% -46% 16%;
      height: 220px;
      background: repeating-linear-gradient(90deg, transparent 0 22px, rgba(88,242,208,.12) 22px 23px);
      transform: rotate(-6deg);
      opacity: .72;
    }}
    h1 {{ font-size: clamp(2.3rem, 6vw, 5.6rem); line-height: .98; margin: 0 0 18px; max-width: 960px; }}
    .hero p {{ color: var(--muted); max-width: 830px; font-size: 1.04rem; line-height: 1.65; }}
    .hero a, .link-row a {{ color: var(--active); font-weight: 800; text-decoration: none; }}
    .link-row {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 18px; }}
    .link-row a {{ border: 1px solid var(--line); border-radius: 999px; padding: 8px 12px; background: rgba(255,255,255,.045); }}
    .floating-nav {{
      position: fixed;
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%);
      z-index: 30;
      width: max-content;
      max-width: calc(100vw - 24px);
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(8, 14, 10, .92);
      box-shadow: 0 18px 60px rgba(0,0,0,.42);
      backdrop-filter: blur(14px);
    }}
    .floating-nav a,
    .floating-nav button {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255,255,255,.055);
      color: var(--active);
      font: inherit;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
    }}
    .floating-nav button {{ color: #061208; background: var(--active); }}
    .floating-status {{ color: var(--muted); font-size: .86rem; min-width: 150px; text-align: center; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; margin: 24px 0 34px; }}
    .metric {{ padding: 18px; border: 1px solid var(--line); border-radius: 22px; background: rgba(255,255,255,.035); }}
    .metric strong {{ display: block; font-size: 1.8rem; letter-spacing: -0.04em; }}
    .metric span {{ color: var(--muted); font-size: .86rem; }}
    .metric.warn strong {{ color: var(--bad); }}
    .section-title {{ margin: 42px 0 16px; display: flex; align-items: end; justify-content: space-between; gap: 20px; }}
    h2 {{ margin: 0; font-size: clamp(1.6rem, 4vw, 3rem); }}
    .note {{ color: var(--muted); max-width: 680px; line-height: 1.55; }}
    .viz-grid {{ display: grid; gap: 18px; }}
    .viz-card {{
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 22px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      background: var(--panel);
      box-shadow: inset 0 1px rgba(255,255,255,.08);
    }}
    .eyebrow {{ color: var(--active); text-transform: uppercase; font-size: .72rem; margin: 0 0 10px; }}
    h3 {{ margin: 0 0 12px; font-size: 2rem; }}
    .small {{ color: var(--muted); font-size: .9rem; line-height: 1.5; }}
    .layer-viz {{ width: 100%; min-height: 220px; border-radius: 24px; background: radial-gradient(circle at 50% 45%, rgba(88,242,208,.12), transparent 58%); }}
    .layer-shell {{ fill: rgba(255,255,255,.045); stroke: rgba(238,248,237,.22); }}
    .active-fill {{ fill: rgba(88,242,208,.72); filter: url(#softGlow); }}
    .frozen-fill {{ fill: rgba(243,180,91,.50); }}
    .neural-line {{ stroke: rgba(175, 255, 220, .75); stroke-width: 1; }}
    .layer-label, .layer-pct {{ fill: var(--text); text-anchor: middle; font-size: 13px; font-weight: 700; }}
    .layer-pct {{ fill: var(--muted); font-size: 11px; }}
    .movie-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .movie-card {{
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 20px;
      background: rgba(14, 22, 18, 0.86);
      display: grid;
      grid-template-columns: 190px 1fr;
      gap: 16px;
      align-items: center;
    }}
    .freeze-movie {{ width: 100%; border-radius: 22px; background: radial-gradient(circle at 50% 46%, rgba(88,242,111,.12), rgba(74,168,255,.06) 62%); }}
    .movie-title {{ fill: var(--text); font-size: 18px; font-weight: 800; }}
    .movie-step {{ fill: var(--muted); font-size: 10px; text-anchor: end; }}
    .movie-edge {{ stroke: rgba(185,255,207,.78); stroke-width: 1.1; }}
    .movie-node {{ stroke: rgba(255,255,255,.72); stroke-width: 1; filter: drop-shadow(0 0 5px rgba(88,242,111,.28)); }}
    .movie-layer {{ fill: var(--muted); text-anchor: middle; font-size: 10px; font-weight: 800; }}
    .movie-freeze-pct {{ fill: var(--muted); text-anchor: middle; font-size: 9px; font-weight: 800; }}
    .spark {{ width: 100%; max-width: 220px; margin-top: 10px; }}
    .spark polyline {{ fill: none; stroke: var(--active); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
    .spark-guide {{ stroke: rgba(255,255,255,.1); stroke-width: 1; }}
    .outcome-card {{
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 16px;
      background:
        linear-gradient(135deg, rgba(88,242,208,.08), transparent 42%),
        rgba(14, 22, 18, 0.86);
      overflow-x: auto;
    }}
    .outcome-viz {{ width: 100%; display: block; }}
    .outcome-title {{ fill: var(--text); font-size: 28px; font-weight: 800; letter-spacing: -0.04em; }}
    .outcome-subtitle, .axis-label, .row-detail {{ fill: var(--muted); font-size: 13px; }}
    .axis-label {{ text-transform: uppercase; letter-spacing: .12em; font-weight: 800; }}
    .row-label {{ fill: var(--text); font-size: 14px; font-weight: 800; }}
    .outcome-track {{ fill: rgba(255,255,255,.055); stroke: rgba(255,255,255,.08); }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 24px; background: var(--panel-2); }}
    .table-wrap.compact table {{ min-width: 460px; }}
    .speed-card {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
      align-items: start;
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      background: linear-gradient(135deg, rgba(74,168,255,.10), rgba(88,242,111,.05));
    }}
    .speed-card p {{ color: var(--muted); line-height: 1.55; }}
    .ladder-card {{
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      margin-top: 22px;
      background: linear-gradient(135deg, rgba(88,242,208,.10), rgba(243,180,91,.05));
    }}
    .ladder-card .table-wrap {{ margin-top: 18px; }}
    .compare-card {{
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      margin-top: 22px;
      background: linear-gradient(135deg, rgba(243,180,91,.08), rgba(88,242,208,.05));
    }}
    .compare-controls {{
      display: grid;
      grid-template-columns: 170px minmax(240px, 420px);
      gap: 12px;
      align-items: center;
      margin: 18px 0;
    }}
    .compare-controls label {{ color: var(--muted); font-weight: 800; text-transform: uppercase; font-size: .78rem; }}
    .compare-controls.two {{ grid-template-columns: 120px minmax(220px, 1fr) 120px minmax(220px, 1fr); }}
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(0,0,0,.28);
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }}
    .branch-chart-wrap {{ border: 1px solid var(--line); border-radius: 24px; background: rgba(0,0,0,.25); overflow-x: auto; margin: 18px 0; }}
    .branch-chart {{ width: 100%; min-width: 920px; display: block; }}
    .branch-overview-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 14px; margin: 18px 0 24px; }}
    .branch-overview-card {{ border: 1px solid var(--line); border-radius: 18px; padding: 14px; background: rgba(255,255,255,.035); }}
    .branch-overview-head {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }}
    .branch-overview-head strong {{ overflow-wrap: anywhere; }}
    .branch-overview-card p {{ color: var(--muted); line-height: 1.45; min-height: 3.9em; }}
    .branch-mini {{ width: 100%; display: block; margin: 6px 0 10px; }}
    .branch-mini rect {{ fill: rgba(0,0,0,.18); stroke: rgba(255,255,255,.08); }}
    .branch-mini line {{ stroke: rgba(255,255,255,.16); }}
    .branch-mini polyline {{ fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
    .branch-mini text {{ fill: var(--muted); text-anchor: middle; font-size: 13px; font-weight: 800; }}
    .branch-panels {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin: 18px 0; }}
    .branch-detail {{ border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: rgba(255,255,255,.035); }}
    .branch-delta {{ margin: -4px 0 18px; }}
    .branch-detail p {{ color: var(--muted); line-height: 1.5; }}
    .branch-stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }}
    .branch-stat {{ border: 1px solid var(--line); border-radius: 12px; padding: 10px; background: rgba(0,0,0,.18); }}
    .branch-stat strong {{ display: block; color: var(--text); }}
    .branch-stat span {{ color: var(--muted); font-size: .82rem; }}
    .branch-axis {{ stroke: rgba(255,255,255,.26); stroke-width: 1; }}
    .branch-grid {{ stroke: rgba(255,255,255,.08); stroke-width: 1; }}
    .branch-line {{ fill: none; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }}
    .branch-chart text {{ fill: var(--muted); font-size: 12px; font-weight: 700; }}
    .branch-chart .chart-title {{ fill: var(--text); font-size: 20px; font-weight: 900; }}
    .branch-tree-card {{
      margin-top: 22px;
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      background: rgba(14, 22, 18, 0.86);
    }}
    .branch-tree {{
      margin: 18px 0 0;
      padding: 18px;
      overflow-x: auto;
      border-radius: 18px;
      background: rgba(0,0,0,.28);
      color: var(--text);
      line-height: 1.55;
      font-size: .92rem;
    }}
    .git-graph {{
      margin: 18px 0;
      padding: 18px;
      overflow-x: auto;
      border-radius: 18px;
      background: rgba(0,0,0,.34);
      color: #dce8ff;
      line-height: 1.45;
      font-size: .84rem;
    }}
    .branch-inventory {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 10px; margin-top: 18px; }}
    .branch-chip {{ border: 1px solid var(--line); border-radius: 14px; padding: 12px; background: rgba(255,255,255,.035); }}
    .branch-chip strong {{ display: block; margin: 8px 0 4px; }}
    .branch-chip small {{ color: var(--muted); line-height: 1.4; }}
    details {{ margin-top: 18px; }}
    summary {{ cursor: pointer; color: var(--active); font-weight: 800; margin-bottom: 12px; }}
    .optional-section {{ margin-top: 42px; border: 1px solid var(--line); border-radius: 24px; padding: 18px; background: rgba(14, 22, 18, 0.70); }}
    .optional-section > summary {{ font-size: clamp(1.25rem, 3vw, 2rem); color: var(--text); }}
    .optional-section .viz-grid {{ margin-top: 16px; }}
    .run-ledger {{ display: grid; gap: 10px; margin-top: 12px; }}
    .run-row {{ display: grid; grid-template-columns: minmax(220px, .85fr) minmax(360px, 1fr) minmax(320px, 1.15fr); gap: 14px; align-items: center; border: 1px solid var(--line); border-radius: 16px; padding: 14px; background: rgba(255,255,255,.032); }}
    .run-main {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; min-width: 0; }}
    .run-main strong {{ font-size: 1rem; overflow-wrap: anywhere; }}
    .run-main span:not(.status) {{ color: var(--muted); }}
    .run-metrics {{ display: grid; grid-template-columns: repeat(4, minmax(70px, 1fr)); gap: 8px; }}
    .run-metrics span {{ border: 1px solid rgba(255,255,255,.08); border-radius: 12px; padding: 8px; background: rgba(0,0,0,.16); }}
    .run-metrics b {{ display: block; font-size: .95rem; }}
    .run-metrics small {{ color: var(--muted); font-size: .72rem; text-transform: uppercase; }}
    .run-paths {{ display: grid; gap: 4px; min-width: 0; }}
    .run-paths span {{ color: var(--muted); }}
    .run-paths code {{ color: var(--text); white-space: normal; overflow-wrap: anywhere; font-size: .88rem; }}
    .status {{
      display: inline-block;
      padding: 1px 7px;
      border-radius: 999px;
      font-size: .72rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .status.done {{ color: #061208; background: var(--good); }}
    .status.validated {{ color: #061208; background: var(--good); }}
    .status.running {{ color: #07121a; background: var(--frozen); }}
    .status.baseline {{ color: #1b1304; background: #f3b45b; }}
    .status.impl {{ color: #07121a; background: var(--frozen); }}
    .status.telemetry {{ color: #061208; background: #58f2d0; }}
    .status.archive {{ color: #150d02; background: #d6b66f; }}
    .status.experiment {{ color: var(--text); background: rgba(255,255,255,.14); }}
    .status.branch {{ color: var(--muted); background: rgba(255,255,255,.08); }}
    .status.blocked {{ color: #220707; background: var(--bad); }}
    .status.todo {{ color: var(--muted); background: rgba(255,255,255,.08); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 0; table-layout: auto; font-size: .92rem; }}
    th, td {{ text-align: left; padding: 10px 11px; border-bottom: 1px solid rgba(255,255,255,.07); white-space: normal; overflow-wrap: anywhere; }}
    th {{ color: var(--muted); font-size: .72rem; text-transform: uppercase; }}
    td {{ font-variant-numeric: tabular-nums; }}
    .good {{ color: var(--good); }}
    .bad {{ color: var(--bad); }}
    .neutral {{ color: var(--muted); }}
    .empty-viz {{ color: var(--muted); padding: 38px; border: 1px dashed var(--line); border-radius: 20px; }}
    @media (max-width: 820px) {{
      main {{ width: min(100vw - 20px, 1760px); padding-top: 20px; }}
      .hero {{ padding: 24px; border-radius: 24px; }}
      .floating-nav {{ bottom: 8px; border-radius: 16px; }}
      .metrics {{ grid-template-columns: repeat(2, 1fr); }}
      .viz-card {{ grid-template-columns: 1fr; }}
      .movie-grid {{ grid-template-columns: 1fr; }}
      .movie-card {{ grid-template-columns: 1fr; }}
      .speed-card {{ grid-template-columns: 1fr; }}
      .compare-controls.two {{ grid-template-columns: 1fr; }}
      .branch-panels {{ grid-template-columns: 1fr; }}
      .run-row {{ grid-template-columns: 1fr; }}
      .run-metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .section-title {{ display: block; }}
    }}
  </style>
</head>
<body>
  <main id="top">
    <section class="hero">
      <p class="eyebrow">Unified STF report from discovered Vast logs only</p>
      <h1>{hero_title}</h1>
      <p>{hero_body}</p>
    <div class="metrics">
        <div class="metric"><strong>{len(completed)}</strong><span>completed variants</span></div>
        <div class="metric"><strong>{branch_count}</strong><span>branches compared</span></div>
        <div class="metric"><strong>{ten_k_completed}</strong><span>completed 10k variants</span></div>
        <div class="metric"><strong>{pct(max_logical_frozen)}</strong><span>max logical freeze</span></div>
        <div class="metric warn"><strong>{pct(max_actual_skip)}</strong><span>actual compute skipped</span></div>
      </div>
      <p>{'Best run: <b>' + html.escape(best.branch) + ' / ' + html.escape(best.score_fn) + '</b> at <b>' + fmt(best.final_bpb) + ' BPB</b>.' if best else ''} Learned-gate is shown separately as a gate-open proxy, because a near-1.0 gate means it is mostly <i>not</i> freezing even when the score looks healthy.</p>
      <div class="link-row">
        <a href="#branch-tree">Branch tree</a>
        <a href="#validation-ladder">Validation ladder</a>
        <a href="#all-run-inventory">All runs</a>
        <a href="#branch-explorer">Branch explorer</a>
        <a href="#reality-check">Reality table</a>
      </div>
    </section>

    {branch_tree}

    {validation_ladder}

    {all_run_inventory}

    {branch_compare}

    <div class="section-title">
      <h2>Freeze Mini Movies</h2>
      <p class="note">Each card is a tiny loop built from the pod telemetry, not a hand-drawn guess. It replays token nodes moving from active green toward frozen blue across training.</p>
    </div>
    <div class="movie-grid">
      {''.join(movie_cards)}
    </div>

    <div class="section-title" id="reality-check">
      <h2>Non-Technical Outcome View</h2>
      <p class="note">This is the “single image” view: each row combines three things people care about without needing model-training jargon. Longer bars are better: lower validation BPB, lower training loss, and lower wall-clock iteration time. Wall-clock speed is not proof of token compute skipping; the compute columns below answer that.</p>
    </div>
    <div class="outcome-card">
      {outcome_dashboard}
    </div>

    <div class="section-title">
      <h2>Speed Reality</h2>
      <p class="note">Freezing should make later layers faster only when frozen tokens are actually skipped in compute, not merely marked frozen in bookkeeping.</p>
    </div>
    {speed_reality}

    <div class="section-title">
      <h2>Reality Check Table</h2>
      <p class="note">Delta is against the branch-local `{html.escape(baseline)}` baseline. Positive delta means better BPB than that branch's baseline.</p>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th><th>Branch</th><th>Formula</th><th>Final BPB</th><th>Delta</th><th>Train Loss</th><th>Avg Iter</th><th>Frozen</th><th>Active</th><th>Compute Mode</th><th>Computed</th><th>Actual Skip</th><th>Skip Eff.</th><th>Control</th><th>Health</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
  </main>
  <nav class="floating-nav" aria-label="Report navigation">
    <a href="#top">Top</a>
    <a href="#branch-tree">Branch tree</a>
    <a href="#validation-ladder">Validation ladder</a>
    <a href="#all-run-inventory">All runs</a>
    <a href="#branch-explorer">Branch explorer</a>
    <a href="#reality-check">Reality table</a>
    <button type="button" id="refreshReportButton" title="Runs the Python summarizer through the local report server">Refresh data</button>
    <span class="floating-status" id="refreshReportStatus">runs Python via server</span>
  </nav>
  <script>
    (function() {{
      const button = document.getElementById("refreshReportButton");
      const status = document.getElementById("refreshReportStatus");
      if (!button || !status) return;
      button.addEventListener("click", async () => {{
        button.disabled = true;
        status.textContent = "Refreshing...";
        try {{
          const refreshUrl = window.location.protocol === "file:"
            ? "http://127.0.0.1:8765/refresh"
            : "/refresh";
          const response = await fetch(refreshUrl, {{ method: "POST" }});
          if (!response.ok) throw new Error(await response.text());
          status.textContent = "Updated, reloading...";
          window.location.reload();
        }} catch (error) {{
          status.textContent = window.location.protocol === "file:"
            ? "Start once: python3 scripts/open_stf_report.py"
            : "Refresh failed";
          console.error(error);
        }} finally {{
          button.disabled = false;
        }}
      }});
    }})();

    (function() {{
      const dataEl = document.getElementById("branchCompareData");
      const defaultsEl = document.getElementById("branchCompareDefaults");
      const selectA = document.getElementById("branchASelect");
      const selectB = document.getElementById("branchBSelect");
      const body = document.getElementById("branchCompareRows");
      const chart = document.getElementById("branchCompareChart");
      const detailA = document.getElementById("branchADetail");
      const detailB = document.getElementById("branchBDetail");
      const deltaDetail = document.getElementById("branchDeltaDetail");
      if (!dataEl || !selectA || !selectB || !body || !chart || !detailA || !detailB || !deltaDetail) return;
      const branches = JSON.parse(dataEl.textContent || "[]");
      const defaults = JSON.parse((defaultsEl && defaultsEl.textContent) || "{{}}");
      const byName = new Map(branches.map((row) => [row.branch, row]));
      if (defaults.a && byName.has(defaults.a)) selectA.value = defaults.a;
      if (defaults.b && byName.has(defaults.b)) selectB.value = defaults.b;
      const fmt = (value, digits = 4) => Number.isFinite(value) ? value.toFixed(digits) : "-";
      const pctValue = (value) => Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "-";
      const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));

      function bestVariant(branch) {{
        if (!branch || !Array.isArray(branch.variants)) return null;
        return branch.variants.slice().sort((a, b) => (a.final_bpb ?? Infinity) - (b.final_bpb ?? Infinity))[0] || null;
      }}

      function selectedBranches() {{
        const first = byName.get(selectA.value) || branches[0];
        const second = byName.get(selectB.value) || branches[1] || first;
        return [first, second].filter((branch, index, list) => branch && list.findIndex((item) => item.branch === branch.branch) === index);
      }}

      function renderDetail(branch, label) {{
        if (!branch) return "";
        const best = bestVariant(branch);
        return `<span class="status ${{esc(branch.status)}}">${{esc(label)}}</span>
          <h3>${{esc(branch.branch)}}</h3>
          <p>${{esc(branch.description)}}</p>
          <div class="branch-stats">
            <div class="branch-stat"><strong>${{esc(best ? best.formula : "-")}}</strong><span>best formula</span></div>
            <div class="branch-stat"><strong>${{fmt(best ? best.final_bpb : null)}}</strong><span>final BPB</span></div>
            <div class="branch-stat"><strong>${{best && best.steps ? best.steps.toLocaleString() : "-"}}</strong><span>iterations</span></div>
            <div class="branch-stat"><strong>${{fmt(best ? best.step_ms : null, 1)}}ms</strong><span>avg iteration</span></div>
            <div class="branch-stat"><strong>${{pctValue(best ? best.actual_skip : null)}}</strong><span>actual skip</span></div>
            <div class="branch-stat"><strong>${{pctValue(best ? best.computed : null)}}</strong><span>computed</span></div>
          </div>`;
      }}

      function renderRows(selected) {{
        const rows = selected.flatMap((branch) =>
          (branch.variants || []).map((variant) => ({{ branch: branch.branch, ...variant }}))
        ).sort((a, b) => (a.final_bpb ?? Infinity) - (b.final_bpb ?? Infinity));
        body.innerHTML = rows.map((row) => `<tr>
          <td>${{esc(row.branch)}}</td>
          <td>${{esc(row.formula)}}</td>
          <td>${{row.steps ? row.steps.toLocaleString() : "-"}}</td>
          <td>${{fmt(row.final_bpb)}}</td>
          <td>${{fmt(row.last_val_bpb)}}</td>
          <td>${{fmt(row.step_ms, 1)}}ms</td>
          <td>${{pctValue(row.actual_skip)}}</td>
          <td>${{pctValue(row.computed)}}</td>
          <td>${{esc(row.source)}}</td>
        </tr>`).join("");
      }}

      function renderDelta(first, second) {{
        const a = bestVariant(first);
        const b = bestVariant(second);
        if (!first || !second || !a || !b) {{
          deltaDetail.innerHTML = "<p>Pick two branches to compare their best available variants.</p>";
          return;
        }}
        const bpbDelta = Number.isFinite(a.final_bpb) && Number.isFinite(b.final_bpb) ? b.final_bpb - a.final_bpb : null;
        const speedDelta = Number.isFinite(a.step_ms) && Number.isFinite(b.step_ms) ? (b.step_ms - a.step_ms) / a.step_ms : null;
        const skipDelta = Number.isFinite(a.actual_skip) && Number.isFinite(b.actual_skip) ? b.actual_skip - a.actual_skip : null;
        deltaDetail.innerHTML = `<span class="status branch">best-to-best</span>
          <h3>${{esc(second.branch)}} vs ${{esc(first.branch)}}</h3>
          <p>Compared by each branch's best final BPB row: ${{esc(second.branch)}} / ${{esc(b.formula)}} against ${{esc(first.branch)}} / ${{esc(a.formula)}}.</p>
          <div class="branch-stats">
            <div class="branch-stat"><strong>${{fmt(bpbDelta)}}</strong><span>BPB delta, lower is better</span></div>
            <div class="branch-stat"><strong>${{pctValue(speedDelta)}}</strong><span>iteration-time delta, lower is faster</span></div>
            <div class="branch-stat"><strong>${{pctValue(skipDelta)}}</strong><span>actual-skip delta, higher is more skipped compute</span></div>
            <div class="branch-stat"><strong>${{fmt(b.final_bpb)}} / ${{fmt(a.final_bpb)}}</strong><span>final BPB pair</span></div>
          </div>`;
      }}

      function renderChart(selected) {{
        const colors = ["#58f26f", "#4aa8ff"];
        const series = [];
        selected.forEach((branch, branchIndex) => {{
          (branch.variants || []).forEach((variant, variantIndex) => {{
            const points = (variant.history || []).map((point) => ({{ step: Number(point.step), bpb: Number(point.bpb) }})).filter((point) => Number.isFinite(point.step) && Number.isFinite(point.bpb));
            if (points.length) series.push({{ branch: branch.branch, formula: variant.formula, color: colors[branchIndex % colors.length], opacity: variantIndex === 0 ? 0.95 : 0.42, points }});
          }});
        }});
        if (!series.length) {{
          chart.innerHTML = '<text x="70" y="210" class="chart-title">No validation history found for the selected branch pair.</text>';
          return;
        }}
        const width = 1120;
        const height = 420;
        const plot = {{ left: 72, top: 58, right: 34, bottom: 58 }};
        const xMax = Math.max(...series.flatMap((item) => item.points.map((point) => point.step)));
        const yValues = series.flatMap((item) => item.points.map((point) => point.bpb));
        const yMinRaw = Math.min(...yValues);
        const yMaxRaw = Math.max(...yValues);
        const yPad = Math.max(0.002, (yMaxRaw - yMinRaw) * 0.12);
        const yMin = yMinRaw - yPad;
        const yMax = yMaxRaw + yPad;
        const x = (step) => plot.left + (step / Math.max(1, xMax)) * (width - plot.left - plot.right);
        const y = (bpb) => plot.top + ((yMax - bpb) / Math.max(0.0001, yMax - yMin)) * (height - plot.top - plot.bottom);
        const parts = [
          `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="rgba(0,0,0,.08)" />`,
          `<text x="72" y="34" class="chart-title">Validation BPB curves for selected branches</text>`,
          `<line x1="${{plot.left}}" y1="${{height - plot.bottom}}" x2="${{width - plot.right}}" y2="${{height - plot.bottom}}" class="branch-axis" />`,
          `<line x1="${{plot.left}}" y1="${{plot.top}}" x2="${{plot.left}}" y2="${{height - plot.bottom}}" class="branch-axis" />`
        ];
        for (let i = 0; i <= 4; i += 1) {{
          const gy = plot.top + ((height - plot.top - plot.bottom) * i / 4);
          const value = yMax - ((yMax - yMin) * i / 4);
          parts.push(`<line x1="${{plot.left}}" y1="${{gy}}" x2="${{width - plot.right}}" y2="${{gy}}" class="branch-grid" />`);
          parts.push(`<text x="16" y="${{gy + 4}}">${{fmt(value, 3)}}</text>`);
        }}
        for (let i = 0; i <= 4; i += 1) {{
          const gx = plot.left + ((width - plot.left - plot.right) * i / 4);
          const value = Math.round(xMax * i / 4);
          parts.push(`<line x1="${{gx}}" y1="${{plot.top}}" x2="${{gx}}" y2="${{height - plot.bottom}}" class="branch-grid" />`);
          parts.push(`<text x="${{gx - 16}}" y="${{height - 24}}">${{value.toLocaleString()}}</text>`);
        }}
        series.forEach((item) => {{
          const pointString = item.points.map((point) => `${{x(point.step).toFixed(1)}},${{y(point.bpb).toFixed(1)}}`).join(" ");
          parts.push(`<polyline class="branch-line" points="${{pointString}}" stroke="${{item.color}}" opacity="${{item.opacity}}" />`);
        }});
        selected.forEach((branch, index) => {{
          parts.push(`<circle cx="${{760}}" cy="${{26 + index * 22}}" r="6" fill="${{colors[index % colors.length]}}" />`);
          parts.push(`<text x="776" y="${{31 + index * 22}}">${{esc(branch.branch)}}</text>`);
        }});
        parts.push(`<text x="${{width - 140}}" y="${{height - 24}}">step</text>`);
        parts.push(`<text x="14" y="54">BPB</text>`);
        chart.innerHTML = parts.join("");
      }}

      function render() {{
        const selected = selectedBranches();
        detailA.innerHTML = renderDetail(byName.get(selectA.value) || selected[0], "Branch A");
        detailB.innerHTML = renderDetail(byName.get(selectB.value) || selected[1] || selected[0], "Branch B");
        renderDelta(byName.get(selectA.value) || selected[0], byName.get(selectB.value) || selected[1] || selected[0]);
        renderRows(selected);
        renderChart(selected);
      }}
      selectA.addEventListener("change", render);
      selectB.addEventListener("change", render);
      render();
    }})();
  </script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=8, help="number of healthy completed variants to scale")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--telemetry", type=int, default=1)
    parser.add_argument("--baseline", default="l2", help="formula used for branch-local comparisons")
    parser.add_argument("--html", type=Path, help="optional path to write an HTML freeze visualization report")
    args = parser.parse_args()

    variants = parse_logs(args.logs)
    completed = dedupe_variants([v for v in variants if v.complete and not v.failed])
    completed.sort(key=lambda v: (v.final_bpb if v.final_bpb is not None else float("inf"), v.last_val_bpb or float("inf")))
    healthy = [v for v in completed if v.health_reason() == "ok"]
    rejected = [v for v in completed if v.health_reason() != "ok"]
    selected = healthy[: args.top]

    print_table("Completed variants ranked by final_bpb", completed)
    print_baseline_comparison(completed, args.baseline)
    if rejected:
        print_table("Completed but filtered out by STF health checks", rejected)
    print_table(f"Selected top {len(selected)} for {args.iterations} iterations", selected)
    emit_commands(selected, args.iterations, args.val_every, args.telemetry)
    if args.html:
        write_html_report(completed, selected, args.html, args.baseline)
        print(f"Wrote HTML report: {args.html}")


if __name__ == "__main__":
    main()
