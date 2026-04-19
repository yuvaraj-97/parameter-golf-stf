# Unified STF Formula Plan For Vast.ai

## Current Decision

Use Vast.ai going forward, not RunPod. The Vast pod should keep a persistent 50GB volume with the full git repo, dataset cache, tokenizer files, logs, telemetry, and experiment archives.

The current STF branches should not be scaled unchanged. The next useful experiment is a serial formula sweep with Telegram lifecycle alerts.

## Formula Variants

| Formula | Env value | Status |
|---|---|---|
| Absolute L2 | `l2` | Control/current behavior |
| Relative L2 | `relative_l2` | First priority |
| Cosine direction change | `cosine` | First priority |
| Direction/projection score | `direction` | First priority |
| Attention/MLP residual split | `attn_mlp_split` | Later, needs extra implementation |
| Loss proxy/token confidence | `loss_proxy` | Later, needs extra implementation |

The runner defaults to only the implemented score names:

```text
l2 relative_l2 cosine direction
```

## Vast.ai Setup

Expected persistent repo layout:

```text
parameter-golf-stf/
  .env
  data/datasets/fineweb10B_sp1024/
  data/tokenizers/fineweb_1024_bpe.model
  logs/
  logs/telemetry/
  vast/experiments/
```

The `.env` file should contain:

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=...
```

## Alert Test

Before implementing `STF_SCORE_FN`, run this to intentionally trigger the preflight failure and verify the 3 STOPPED alerts:

```bash
STF_BRANCHES="stf-learned-gate" \
ITERATIONS=1 \
bash scripts/run_vast_formula_series.sh
```

Expected behavior:

- one STARTED series Telegram message
- checkout of `stf-learned-gate`
- failure because `train_gpt.py` does not yet contain `stf_score_fn`
- three STOPPED Telegram messages

## First Real Smoke Run

After `STF_SCORE_FN` is implemented on the target branch:

```bash
STF_BRANCHES="stf-learned-gate" \
STF_SCORE_FNS="l2 relative_l2 cosine direction" \
ITERATIONS=500 \
VAL_LOSS_EVERY=50 \
STF_TELEMETRY=1 \
WARMUP_STEPS=0 \
MUON_BACKEND_STEPS=3 \
bash scripts/run_vast_formula_series.sh
```

Then repeat with:

```bash
STF_BRANCHES="stf-minimal"
```

## Wider Matrix

Only after 500-step telemetry looks sane:

```bash
STF_BRANCHES="stf-minimal stf-learned-gate stf-soft-freeze stf-reactivation stf-budget-regularization stf-recurrence stf-quantization" \
STF_SCORE_FNS="l2 relative_l2 cosine direction" \
ITERATIONS=2000 \
VAL_LOSS_EVERY=200 \
STF_TELEMETRY=0 \
WARMUP_STEPS=0 \
MUON_BACKEND_STEPS=3 \
bash scripts/run_vast_formula_series.sh
```

## Decision Rules

- `active_mean > 0.95`: threshold is too low for that formula.
- `active_mean < 0.10`: threshold is too high for that formula.
- `score_mean` should be finite and stable.
- `val_bpb` should be finite and not obviously diverging.
- Do not scale a formula to 10k unless it is competitive by 2k/5k.

## Archives

The runner writes local archives to:

```text
vast/experiments/<date>-<gpu>-<gpu_count>gpu/<branch>/<run_id>/
```

Each archive includes train logs, console output, telemetry if available, `train_gpt.py`, and final model artifacts if produced.
