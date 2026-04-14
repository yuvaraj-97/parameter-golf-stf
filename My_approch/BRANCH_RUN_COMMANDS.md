# One-Shot Branch Commands (Train + Telemetry + Save + Push)

Use these on the pod in `/workspace/parameter-golf`.

## One command pattern

```bash
bash scripts/run_train_and_save.sh <branch> <iterations> <run_number>
```

`run_number` is optional and defaults to `1`.

## Baseline

```bash
bash scripts/run_train_and_save.sh baseline-repro 2000 1
bash scripts/run_train_and_save.sh baseline-repro 5000 1
bash scripts/run_train_and_save.sh baseline-repro 20000 1
```

## STF Minimal

```bash
bash scripts/run_train_and_save.sh stf-minimal 2000 1
bash scripts/run_train_and_save.sh stf-minimal 5000 1
bash scripts/run_train_and_save.sh stf-minimal 20000 1
```

## STF Learned Gate

```bash
bash scripts/run_train_and_save.sh stf-learned-gate 2000 1
bash scripts/run_train_and_save.sh stf-learned-gate 5000 1
bash scripts/run_train_and_save.sh stf-learned-gate 20000 1
```

## STF Soft Freeze

```bash
bash scripts/run_train_and_save.sh stf-soft-freeze 2000 1
bash scripts/run_train_and_save.sh stf-soft-freeze 5000 1
bash scripts/run_train_and_save.sh stf-soft-freeze 20000 1
```

## STF Reactivation

```bash
bash scripts/run_train_and_save.sh stf-reactivation 2000 1
bash scripts/run_train_and_save.sh stf-reactivation 5000 1
bash scripts/run_train_and_save.sh stf-reactivation 20000 1
```

## STF Budget Regularization

```bash
bash scripts/run_train_and_save.sh stf-budget-regularization 2000 1
bash scripts/run_train_and_save.sh stf-budget-regularization 5000 1
bash scripts/run_train_and_save.sh stf-budget-regularization 20000 1
```

## STF Recurrence

```bash
bash scripts/run_train_and_save.sh stf-recurrence 2000 1
bash scripts/run_train_and_save.sh stf-recurrence 5000 1
bash scripts/run_train_and_save.sh stf-recurrence 20000 1
```

## STF Quantization

```bash
bash scripts/run_train_and_save.sh stf-quantization 2000 1
bash scripts/run_train_and_save.sh stf-quantization 5000 1
bash scripts/run_train_and_save.sh stf-quantization 20000 1
```

## Optional overrides

```bash
# force single GPU
NPROC_PER_NODE=1 bash scripts/run_train_and_save.sh stf-minimal 2000 1

# change telemetry sampling interval
TELEMETRY_INTERVAL_SECONDS=5 bash scripts/run_train_and_save.sh stf-minimal 2000 1
```

