# STF Branch Run Commands (Copy/Paste)

Use these on the pod in `/workspace/parameter-golf`.

## Shared Assumptions

- Dataset: `./data/datasets/fineweb10B_sp1024/`
- Tokenizer: `./data/tokenizers/fineweb_1024_bpe.model`
- Vocab size: `1024`
- Validation cadence: `200`
- This pod currently has 4x RTX 5090 and all are active.

If you want to keep using all 4 GPUs:
- use `--nproc_per_node=4`

If you want strict single-GPU comparability:
- use `--nproc_per_node=1`

## 1) Baseline (if rerun needed)

```bash
git checkout baseline-repro
export RUN_ID="2026-04-14_baseline-repro_exp1"
export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE="1024"
export VAL_LOSS_EVERY="200"

# terminal A: telemetry
chmod +x scripts/capture_telemetry.sh
./scripts/capture_telemetry.sh
```

```bash
# terminal B: training
git checkout baseline-repro
export RUN_ID="2026-04-14_baseline-repro_exp1"
export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE="1024"
export VAL_LOSS_EVERY="200"
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

```bash
# after training (stop telemetry with Ctrl+C first)
git checkout baseline-repro
export RUN_ID="2026-04-14_baseline-repro_exp1"
bash save.sh
```

## 2) Next Branch: stf-minimal

```bash
git checkout stf-minimal
git pull --ff-only origin stf-minimal
export RUN_ID="2026-04-14_stf-minimal_exp1"
export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE="1024"
export VAL_LOSS_EVERY="200"

# terminal A: telemetry
chmod +x scripts/capture_telemetry.sh
./scripts/capture_telemetry.sh
```

```bash
# terminal B: training
git checkout stf-minimal
export RUN_ID="2026-04-14_stf-minimal_exp1"
export DATA_PATH="./data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE="1024"
export VAL_LOSS_EVERY="200"
torchrun --standalone --nproc_per_node=4 train_gpt.py
```

```bash
# after training
git checkout stf-minimal
export RUN_ID="2026-04-14_stf-minimal_exp1"
bash save.sh
```

## 3) Remaining Branch Sequence

Repeat the same pattern (branch checkout first, unique `RUN_ID`, telemetry in terminal A, training in terminal B, `bash save.sh` at end):

1. `stf-learned-gate`
2. `stf-soft-freeze`
3. `stf-reactivation`
4. `stf-budget-regularization`
5. `stf-recurrence`
6. `stf-quantization`

