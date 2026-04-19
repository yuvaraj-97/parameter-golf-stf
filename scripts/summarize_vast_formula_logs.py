#!/usr/bin/env python3
"""Summarize partial Vast STF formula sweep logs and emit follow-up commands."""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


KEY_VALUE_RE = re.compile(r"([A-Za-z_]+):([-+0-9.eE]+)")
VAL_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([-+0-9.eE]+)\s+val_bpb:([-+0-9.eE]+)")
FINAL_RE = re.compile(r"final_int8_zlib_roundtrip(?:_exact)?\s+val_loss:([-+0-9.eE]+)\s+val_bpb:([-+0-9.eE]+)")


@dataclass
class Variant:
    source: str
    branch: str
    score_fn: str
    run_id: str = ""
    final_loss: float | None = None
    final_bpb: float | None = None
    last_val_step: int | None = None
    last_val_bpb: float | None = None
    stats: dict[str, float] = field(default_factory=dict)
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


def parse_float(value: str) -> float | None:
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


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
                    variant.last_val_bpb = parse_float(val_match.group(4))

                final_match = FINAL_RE.search(line)
                if final_match:
                    variant.final_loss = parse_float(final_match.group(1))
                    variant.final_bpb = parse_float(final_match.group(2))

                if "stf_stats" in line:
                    for key, value in KEY_VALUE_RE.findall(line):
                        parsed = parse_float(value)
                        if parsed is not None:
                            variant.stats[key] = parsed

                if "VARIANT FAILED" in line or "Traceback " in line:
                    variant.failed = True

    return list(variants.values())


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=8, help="number of healthy completed variants to scale")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--telemetry", type=int, default=1)
    parser.add_argument("--baseline", default="l2", help="formula used for branch-local comparisons")
    args = parser.parse_args()

    variants = parse_logs(args.logs)
    completed = [v for v in variants if v.complete and not v.failed]
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


if __name__ == "__main__":
    main()
