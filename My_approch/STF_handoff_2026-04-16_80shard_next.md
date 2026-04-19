# STF Handoff: Current Runbook

This is the handoff file for continuing STF / Parameter Golf work in a fresh session.

Last updated: 2026-04-16

## 1) Current State

Goal: beat the Parameter Golf naive baseline score while staying under the 16 MB artifact cap.

Public target from the repo README:

| Target | Score |
|---|---:|
| Naive Baseline | `1.2244` val_bpb |

Important rule: use `final_int8_zlib_roundtrip_exact`, not only raw validation BPB. The artifact cap is decimal `16,000,000` bytes.

Current conclusion:

- 1-shard runs were useful smoke tests only.
- 80-shard runs are the decision source.
- Current STF branches should not be scaled unchanged because baseline wins on 80 shards.
- The active next run is `baseline-repro 10000 run2` on 80 shards.

## 2) 80-Shard 5k Results

These are the challenge-aligned 5k results using `train_shards:80`.

| Rank | Branch | Roundtrip BPB | Raw BPB | Artifact bytes | 16MB margin | Step avg | Verdict |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `baseline-repro` | `1.25187638` | `1.2475` | `15,845,108` | `154,892` | `389.39ms` | Current winner |
| 2 | `stf-learned-gate` | `1.26287554` | `1.2585` | `15,835,231` | `164,769` | `612.29ms` | Worse quality, slower |
| 3 | `stf-recurrence` | `1.26902905` | `1.2649` | `15,832,462` | `167,538` | `614.64ms` | Worse quality, slower |
| 4 | `stf-soft-freeze` | `1.27656162` | `1.2727` | `15,834,261` | `165,739` | `613.90ms` | Worse quality, slower |

Baseline is still improving at 5k:

| Step | Raw val_bpb |
|---:|---:|
| `1000` | `1.3835` |
| `2000` | `1.3234` |
| `3000` | `1.2989` |
| `4000` | `1.2801` |
| `5000` | `1.2475` |

This supports a 10k baseline run before deciding whether 20k is worth the cost.

## 3) Active Next Run

The user has already started this run:

```bash
cd /workspace/parameter-golf

python3 data/cached_challenge_fineweb.py --variant sp1024

echo "train_shards=$(ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"

bash scripts/run_train_and_save.sh baseline-repro 10000 2
```

Expected preflight:

```text
train_shards=80
```

Do not use `COMPILE_MODEL=1` for this run.

Do not run STF 10k yet.

## 4) Decision Gate After 10k Baseline

When the 10k baseline run finishes and logs are pushed:

```bash
git fetch origin
python scripts/generate_stf_analysis.py 10000 preferred
python scripts/generate_stf_analysis.py 5000 80
```

Then inspect:

```text
My_approch/analysis/stf_10000_branch_compare.html
My_approch/analysis/stf_10000_branch_compare_summary.csv
```

Decision rule:

| 10k baseline roundtrip BPB | Next step |
|---:|---|
| `<= 1.2244` | Run `baseline-repro 20000 2`; start compression cleanup |
| `1.2244` to `1.235` | Run `baseline-repro 20000 2`; still plausible |
| `> 1.235` | Pause scaling; work on architecture/compression/STF formula changes |

Artifact gate:

- If artifact margin drops below `50,000` bytes, pause before 20k and prioritize compression.
- Current 80-shard 5k baseline margin was `154,892` bytes, so there is room but not much.

## 5) STF Research Direction

Current STF score:

```python
delta = torch.linalg.vector_norm((x_out - x_in).float(), dim=-1, ord=2, keepdim=True)
```

This was not competitive on 80 shards. It likely measures raw hidden movement rather than whether a token still needs useful compute.

Next STF experiment should change the score formula before any more scaling.

Recommended order:

1. Normalized residual change:

```text
score = ||x_out - x_in|| / (||x_in|| + eps)
```

2. Directional/cosine change:

```text
score = 1 - cosine_similarity(x_out, x_in)
```

3. Combined magnitude + direction:

```text
score = normalized_residual * directional_change
```

4. Confidence-aware gating:

```text
keep uncertain/high-entropy tokens active
```

First branch to modify: `stf-learned-gate`.

Reason:

- It has soft continuous routing.
- It was the best STF branch by raw BPB on 80 shards.
- It is less brittle than hard-freeze branches.

Do not modify `baseline-repro` for STF experiments.

## 6) Compile Mode Policy

Current default:

```bash
COMPILE_MODEL=0
```

Do not use `COMPILE_MODEL=1` for current 10k baseline or current STF branches.

Why:

- STF quality is currently worse than baseline on 80 shards.
- Compile mode is a speed optimization, not a quality fix.
- Earlier STF compile attempts hit Dynamo/fullgraph issues.
- `train_gpt.py` currently compiles the model with `fullgraph=True`.

Only test compile mode after a branch is quality-competitive.

Compile validation ladder:

```bash
COMPILE_MODEL=1 bash scripts/run_train_and_save.sh <candidate-branch> 200 1
COMPILE_MODEL=1 bash scripts/run_train_and_save.sh <candidate-branch> 2000 1
```

Only if both pass and BPB is unchanged should compile mode be used for 5k/10k/20k.

## 7) PR Tricks To Cherry-Pick Later

Do not add these before the 10k baseline result. After the 10k result, test one controlled idea at a time:

- QK gain tuning.
- Longer context / sliding eval.
- Parallel residuals.
- Depth recurrence as architecture, not just STF carry blending.
- Compression upgrades: zstd, int6/mixed int6-int8, GPTQ-lite, QAT.

Compression matters because the current artifact is already close to 16 MB.

## 8) Useful Analysis Commands

Regenerate the current reports:

```bash
python scripts/generate_stf_analysis.py 2000 preferred
python scripts/generate_stf_analysis.py 5000 1
python scripts/generate_stf_analysis.py 5000 80
```

Start the local analysis server:

```bash
python scripts/serve_stf_analysis.py
```

Open:

```text
http://127.0.0.1:8765/
```

Useful report files:

```text
My_approch/analysis/stf_5000_shards80_branch_compare.html
My_approch/analysis/stf_5000_shards80_branch_compare_summary.csv
My_approch/analysis/stf_5000_shards1_branch_compare.html
My_approch/analysis/stf_2000_branch_compare.html
```

## 9) Pod Run Checklist

Before any meaningful run:

```bash
cd /workspace/parameter-golf
git branch --show-current
git pull --ff-only origin "$(git branch --show-current)"
python3 data/cached_challenge_fineweb.py --variant sp1024
echo "train_shards=$(ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"
df -h /workspace
```

Expected:

```text
train_shards=80
```

If it says `1`, stop. That is only the debug dataset.

## 10) What Not To Do Next

- Do not run current STF branches at 10k or 20k unchanged.
- Do not interpret 1-shard wins as challenge progress.
- Do not enable compile mode before a branch is quality-competitive.
- Do not compare raw BPB without checking final int8/zlib roundtrip BPB.
- Do not ignore artifact size; the cap is `16,000,000` bytes.
