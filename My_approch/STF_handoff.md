# STF Handoff: Branch-Safe Runbook

This is the one file to use when starting a fresh session for STF work.

## 1) Quick answers to current blockers

- Why run a baseline if OpenAI already has one?
  - You do not need to reproduce their entire history.
  - You should still run at least one baseline in *your* repo/runtime so your STF deltas are comparable against your own environment and commit.
- Can `val_bpb` go down, then up, then down again?
  - Yes. That is normal. Track `best-so-far`, not only the latest point.

## 2) What "config" means in this repo

`train_gpt.py` is configured via environment variables, for example:
- Data/tokenizer: `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE`
- Run identity: `RUN_ID`
- Eval cadence: `VAL_LOSS_EVERY`
- Training length/cap: `ITERATIONS`, `MAX_WALLCLOCK_SECONDS`
- Model shape: `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`
- Optimization knobs: `EMBED_LR`, `MATRIX_LR`, `BETA1`, `BETA2`, etc.

## 3) Required branch-safe command pattern

Always start with branch checkout first, then set `RUN_ID`.

PowerShell:

```powershell
git checkout <branch>
$env:RUN_ID = "<yyyy-mm-dd>_<branch>_<short-tag>"
```

Bash:

```bash
git checkout <branch>
export RUN_ID="<yyyy-mm-dd>_<branch>_<short-tag>"
```

## 4) Baseline command (copy/paste)

PowerShell:

```powershell
git checkout baseline-repro
$env:RUN_ID = "2026-04-14_baseline-repro_smoke1"
$env:DATA_PATH = "./data/datasets/fineweb10B_sp1024/"
$env:TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
$env:VOCAB_SIZE = "1024"
$env:VAL_LOSS_EVERY = "200"
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Linux/bash:

```bash
git checkout baseline-repro
export RUN_ID="2026-04-14_baseline-repro_smoke1"
export DATA_PATH=./data/datasets/fineweb10B_sp1024/
export TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export VAL_LOSS_EVERY=200
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## 5) STF minimal command template

Use this template on STF branches (replace branch/tag). Keep branch checkout at top.

PowerShell:

```powershell
git checkout stf-minimal
$env:RUN_ID = "2026-04-14_stf-minimal_exp1"
$env:DATA_PATH = "./data/datasets/fineweb10B_sp1024/"
$env:TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
$env:VOCAB_SIZE = "1024"
$env:VAL_LOSS_EVERY = "200"
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## 6) Log capture and push from pod

Use the repo-root `save.sh` script after a run finishes:

```bash
git checkout <branch>
export RUN_ID="<same-run-id-used-in-training>"
bash save.sh
```

`save.sh` will:
- copy `logs/$RUN_ID.txt` to a structured run folder
- copy optional `final_model.pt` and `final_model.int8.ptz`
- write a README with GPU/pod/branch metadata and final metric line
- commit and push to the current branch

## 7) Suggested branch execution order

1. `baseline-repro`
2. `stf-minimal`
3. `stf-learned-gate`
4. `stf-soft-freeze`
5. `stf-reactivation`
6. `stf-budget-regularization`
7. `stf-recurrence`
8. `stf-quantization`

## 8) Pre-flight checks before any run

PowerShell:

```powershell
git checkout <branch>
git pull origin <branch>
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
Test-Path .\data\datasets\fineweb10B_sp1024
Test-Path .\data\tokenizers\fineweb_1024_bpe.model
```

If these pass, launch training.
