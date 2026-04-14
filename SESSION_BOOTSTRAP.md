# SESSION BOOTSTRAP (STF)

Use this file at the start of any new AI session so context is not lost.

## Canonical Repo

- Primary working repo: `C:\Users\yuvar\OneDrive\Documents\GitHub\parameter-golf-stf`
- This file was generated from a sandboxed session in `parameter-golf`; copy it into `parameter-golf-stf`.

## Goal

- Beat challenge `val_bpb` under Parameter Golf constraints.
- Strategy focus: Selective Token Freezing (STF).

## Current Branch Topology

- `main`
- `baseline-repro`
- `stf-minimal`
- `stf-learned-gate`
- `stf-soft-freeze`
- `stf-reactivation`
- `stf-budget-regularization`
- `stf-recurrence`
- `stf-quantization`

All created and tracking `origin/*`.

## Fixed Decisions (Q1-Q15)

1. Baseline anchor: use one fixed commit hash for all comparisons. Current observed base was `75700cb`; verify before runs.
2. Dataset flow: quick smoke/debug first, then FineWeb challenge path.
3. Primary metric: `val_bpb` (best-so-far), with compute/time context.
4. Freeze warmup: no freezing early, then ramp.
5. Target deep-layer active ratio: roughly 35-55%, avoid collapse.
6. Guardrail: minimum active depth before freezing.
7. Gradient stability: preserve residual/cached path for frozen tokens.
8. Initial freeze signal: normalized `||h_new - h_old||` with smoothing.
9. Collapse detection: independent of best `val_bpb`; keep detector plus early-stop policy.
10. Reactivation: disabled in v1.
11. Required ablations: threshold/warmup/depth cap/seed sweeps.
12. Kill policy: stop a branch early if trend is clearly not competitive vs baseline at same budget slice.
13. Merge policy: no experimental merges to `main` until proven.
14. Metric behavior: `val_bpb` can go down, up, then down again. Use best checkpoint, not latest.
15. Success framing: reproducible challenge improvement with logs and artifacts.

## Why Baseline Run Is Still Needed

- Not to re-prove OpenAI baseline globally.
- To establish *your* runtime/control baseline in your repo, hardware, and commit context for fair delta comparisons.

## Environment Variables Used By `train_gpt.py`

Core run config:
- `RUN_ID`
- `DATA_PATH`
- `TOKENIZER_PATH`
- `VOCAB_SIZE`
- `VAL_LOSS_EVERY`
- `ITERATIONS`
- `MAX_WALLCLOCK_SECONDS`

Model/optimizer knobs are also env-driven (`NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, etc.).

## Mandatory Run Pattern (branch-first + RUN_ID)

PowerShell:

```powershell
git checkout <branch>
$env:RUN_ID = "YYYY-MM-DD_<branch>_<tag>"
$env:DATA_PATH = "./data/datasets/fineweb10B_sp1024/"
$env:TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
$env:VOCAB_SIZE = "1024"
$env:VAL_LOSS_EVERY = "200"
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Do not skip `git checkout <branch>` at the top.

## Standard First Commands In A New Session

```powershell
cd C:\Users\yuvar\OneDrive\Documents\GitHub\parameter-golf-stf
git fetch origin
git checkout main
git pull origin main
git branch -vv
git rev-parse HEAD
```

Record:
- `BASELINE_COMMIT=<hash>`

## Baseline Sanity Run (short, local)

```powershell
git checkout baseline-repro
$env:RUN_ID = "2026-04-14_baseline-repro_smoke1"
$env:DATA_PATH = "./data/datasets/fineweb10B_sp1024/"
$env:TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
$env:VOCAB_SIZE = "1024"
$env:VAL_LOSS_EVERY = "200"
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Run Logging / Artifact Push Workflow

Use repo-root `save.sh` after each run:

```bash
git checkout <branch>
export RUN_ID="<same id used for training>"
bash save.sh
```

Expected files from training:
- required: `logs/<RUN_ID>.txt`
- optional: `final_model.pt`, `final_model.int8.ptz`

`save.sh` behavior:
- creates structured run folder under `runpod/experiments/.../<branch>/<RUN_ID>/`
- copies logs/artifacts
- writes run README with metadata and final metric line
- commits + pushes to current branch

## Execution Order

1. `baseline-repro`
2. `stf-minimal`
3. `stf-learned-gate`
4. `stf-soft-freeze`
5. `stf-reactivation`
6. `stf-budget-regularization`
7. `stf-recurrence`
8. `stf-quantization`

## Branch Hygiene Rules

- Always branch checkout first in every command block.
- Use explicit remote target in push commands: `git push origin <branch>`.
- Keep experiment results in branch-specific commits.
- Do not merge experimental branches to `main` until evidence is solid.

## Known Git/PowerShell Notes

- PowerShell 5.x does not support `&&`; use `;` or separate lines.
- If push fails due to non-fast-forward against a newly initialized GitHub repo, resolve explicitly (fetch/rebase or force-with-lease intentionally).

## Hand-off Prompt For New AI Session

Paste this in the first message of a new session:

> Read `SESSION_BOOTSTRAP.md`, `My_approch/STF_handoff.md`, and `save.sh`.  
> We are in `parameter-golf-stf`.  
> Follow branch-first commands, set `RUN_ID` for every run, and keep logs committed per branch.  
> Start from `baseline-repro` sanity run, then guide me into `stf-minimal`.

