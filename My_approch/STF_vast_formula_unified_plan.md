# Unified STF Formula Plan For Vast.ai

## Current Decision

Use Vast.ai going forward, not RunPod. The Vast pod should keep a persistent 50GB volume with the full git repo, dataset cache, tokenizer files, logs, telemetry, and experiment archives.

The previous 80-shard result says the current STF implementations should not be scaled unchanged:

| Branch | 5k 80-shard result | Decision |
|---|---:|---|
| baseline-repro | 1.25187638 roundtrip BPB | Current winner |
| stf-learned-gate | 1.26287554 roundtrip BPB | Needs formula change first |
| stf-recurrence | 1.26902905 roundtrip BPB | Needs formula change first |
| stf-soft-freeze | 1.27656162 roundtrip BPB | Needs formula change first |

So the next STF work is not "run longer"; it is "run the scoring formulas in a controlled series and compare telemetry plus BPB."

## Formula Variants To Test

These are the variants named across the Claude and Codex plans.

| Formula | Env value | Status | Meaning |
|---|---|---|---|
| Absolute L2 | `l2` | Control | `||x_out - x_in||`; current STF behavior |
| Relative L2 / normalized residual | `relative_l2` | First priority | `||x_out - x_in|| / (||x_in|| + eps)` |
| Cosine direction change | `cosine` | First priority | `(1 - cosine(x_out, x_in)) / 2` |
| Direction/projection score | `direction` | First priority | `abs(cosine(x_out - x_in, x_in))` |
| Attention/MLP residual split | `attn_mlp_split` | Later implementation | Needs block internals split before scoring |
| Loss proxy/token confidence | `loss_proxy` | Later implementation | Needs logits/entropy or margin plumbing |

The serial runner defaults to the implemented score functions only:

```text
l2 relative_l2 cosine direction
```

Do not include `attn_mlp_split` or `loss_proxy` in the run matrix until `train_gpt.py` has real implementations for them.

## Vast.ai Runtime Setup

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

The `.env` file should already contain:

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_USER_ID=...
```

The runner sources `.env` automatically and sends Telegram updates for:

- series start
- variant started
- variant ended
- moving to next variant
- all variants complete
- crash/stop alerts, repeated 3 times

## Recommended Run Matrix

Start with the branch that has the best chance of using a better score softly:

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

Then test the simpler hard-freeze mode:

```bash
STF_BRANCHES="stf-minimal" \
STF_SCORE_FNS="l2 relative_l2 cosine direction" \
ITERATIONS=500 \
VAL_LOSS_EVERY=50 \
STF_TELEMETRY=1 \
WARMUP_STEPS=0 \
MUON_BACKEND_STEPS=3 \
bash scripts/run_vast_formula_series.sh
```

Only after the 500-step telemetry looks sane should the matrix widen to all STF branches:

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

For 500-step smoke runs:

- `active_mean > 0.95`: threshold is too low for that formula.
- `active_mean < 0.10`: threshold is too high for that formula.
- `score_mean` should be finite and stable.
- `val_bpb` should be finite and not obviously diverging.

For 2k/5k validation runs:

- Keep the 2 best formulas by BPB trajectory and active ratio stability.
- Prefer formulas that beat or match L2 at equal branch/iteration count.
- Do not scale a formula to 10k unless it is competitive by 2k/5k.

## Compute Controls

Use these during formula search:

```bash
WARMUP_STEPS=0
MUON_BACKEND_STEPS=3
STF_TELEMETRY=1   # smoke only
STF_TELEMETRY=0   # longer runs
MAX_WALLCLOCK_SECONDS=0
```

This removes avoidable baseline compute from the experiment without changing the main architecture. The largest practical saving is skipping warmup during branch experiments; Muon backend steps are a smaller but useful optimizer-side saving.

## Runner

The serial runner is:

```text
scripts/run_vast_formula_series.sh
```

It does not use RunPod paths. It writes local archives to:

```text
vast/experiments/<date>-<gpu>-<gpu_count>gpu/<branch>/<run_id>/
```

Each archive includes:

- `train.log` if produced by `train_gpt.py`
- `console.log`
- `telemetry.csv` if telemetry capture is available
- `train_gpt.py`
- final model artifacts if produced
- `README.md` with branch, formula, commit, GPU, and final metric line

## Important Preflight

The target STF branches must support:

```bash
STF_SCORE_FN=l2|relative_l2|cosine|direction
```

If a branch has not yet been patched with `STF_SCORE_FN`, the runner stops before training and sends the 3 crash alerts. Patch the score function into the STF branches before running the full matrix.
