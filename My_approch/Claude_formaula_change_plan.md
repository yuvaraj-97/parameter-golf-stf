# STF Analysis & New Scoring Formulas Plan

## Context

The `parameter-golf-stf` repo is an OpenAI challenge where the goal is to minimize `val_bpb` under strict compute/size constraints. The strategy is **Selective Token Freezing (STF)**: skipping or dampening block updates for tokens whose hidden state changes little between layers.

Seven STF branches exist on remote (`stf-minimal`, `stf-learned-gate`, `stf-soft-freeze`, `stf-reactivation`, `stf-budget-regularization`, `stf-recurrence`, `stf-quantization`). They all share **identical `apply_stf()` logic** — the only difference per branch is the default `STF_MODE` env var and a few hyperparameter defaults in `Hyperparameters`.

User's three goals:
1. Understand how token freezing actually works across branches and whether it's real or just overhead
2. Add 3 new scoring formulas (relative-L2, cosine, direction)
3. Find compute (A) to remove from baseline X so total becomes X + Y - A

---

## Part 1: STF Branch Analysis (findings)

### How the Shared `apply_stf()` Works

**File:** `train_gpt.py` (all STF branches), lines ~864–937

```python
def apply_stf(layer_idx, x_in, x_out):
    # Skip first stf_warmup_layers (default 3) and after stf_depth_cap
    delta = torch.linalg.vector_norm((x_out - x_in).float(), dim=-1, ord=2, keepdim=True)
    ema_score = ema_decay * ema_score + (1 - ema_decay) * delta   # cross-layer EMA
    active = ema_score >= threshold   # binary freeze decision
    return active_float * x_out + (1 - active_float) * <frozen_value>
```

The "frozen value" differs by mode:

| Branch | `stf_mode` default | Frozen-token output | Key defaults |
|---|---|---|---|
| `stf-minimal` | `minimal` | `x_in` (identity pass) | threshold=0.045, ema=0.85 |
| `stf-learned-gate` | `learned_gate` | `sigmoid(scale*(ema-thresh)+bias)*x_out + (1-gate)*x_in` | threshold=0.020, trainable gate |
| `stf-soft-freeze` | `soft_freeze` | `soft_keep*x_out + (1-soft_keep)*x_in` | threshold=0.045, soft_keep=0.35 |
| `stf-reactivation` | `reactivation` | `x_in`, but forced active every N layers | threshold=0.045, reactivate_every=2 |
| `stf-budget-regularization` | `budget_regularization` | `x_in`, adaptive threshold | kp=3.0, target_active=0.45 |
| `stf-recurrence` | `recurrence` | `recur_mix*x_out + (1-recur_mix)*x_in` | recur_mix=0.55 |
| `stf-quantization` | `quantization` | `round(x_in * scale) / scale` | quant_scale=64 |

### Is it Real or Just Overhead?

**Currently: it is overhead-only.** The full block forward (attention + MLP) still runs for ALL tokens at every layer. `apply_stf` only blends outputs post-hoc. No tokens are actually skipped from the attention kernel.

The telemetry fields `active_mean` and `active_by_layer` in `stf_stats` log lines show how many tokens would theoretically be frozen — but those tokens still consumed full compute. The session from yesterday was measuring this: the EMA score and threshold are calibrated, so theoretically 45–65% of tokens could be "frozen", yet no FLOP savings occur.

**Y (STF overhead) per layer per step includes:**
- `vector_norm()` over `[B*T, D]`: O(T·D)
- EMA multiply-add over `[B*T, 1]`
- Threshold comparison + mask cast
- Blended output write: O(T·D)
- 9 stat buffers updated per layer (when `STF_TELEMETRY=1`)

---

## Part 2: New Scoring Formulas

### Current formula (only one exists):
```python
delta = torch.linalg.vector_norm((x_out - x_in).float(), dim=-1, ord=2, keepdim=True)
```
This is an **absolute L2 norm** of the change. It doesn't normalize by token size.

### 3 formulas to add via `STF_SCORE_FN` env var:

**A. `relative_l2`** — "variation compared to its size"
```python
delta = torch.linalg.vector_norm((x_out - x_in).float(), dim=-1, ord=2, keepdim=True)
norm_in = torch.linalg.vector_norm(x_in.float(), dim=-1, ord=2, keepdim=True)
score = delta / (norm_in + 1e-6)
```
Interpretation: what fraction of the token's magnitude changed. Tokens with large representations that change little score low.

**B. `cosine`** — angular change between token states
```python
cos_sim = F.cosine_similarity(x_in.float(), x_out.float(), dim=-1, eps=1e-6).unsqueeze(-1)
score = (1.0 - cos_sim) / 2.0   # maps [-1,1] → [0,1], 0=identical direction
```
Interpretation: purely directional. A frozen token's representation didn't rotate. Scale-invariant.

**C. `direction`** — how much the change aligns with the original token direction
```python
delta = (x_out - x_in).float()
cos_proj = F.cosine_similarity(delta, x_in.float(), dim=-1, eps=1e-6).abs().unsqueeze(-1)
score = cos_proj   # high = change aligns with current representation axis
```
Interpretation: tokens that change along their own direction are "growing" (important). Tokens changing orthogonally may be less impactful. Freeze when score is LOW.

> Note: `direction` score is HIGH when token changes in its own axis, LOW when change is perpendicular. All formulas are designed so LOW score → freeze, HIGH score → active.

### Where to implement

**File:** `train_gpt.py` in each STF branch (all share the same `apply_stf` body)

**Add to `Hyperparameters`:**
```python
stf_score_fn = os.environ.get("STF_SCORE_FN", "l2")  # "l2" | "relative_l2" | "cosine" | "direction"
```

**Add to `GPT.__init__` signature + body** (store as `self.stf_score_fn`)

**Replace the `delta` computation block** in `apply_stf()`:
```python
# Was: delta = torch.linalg.vector_norm((x_out - x_in).float(), dim=-1, ord=2, keepdim=True)
x_in_f = x_in.float()
x_out_f = x_out.float()
diff = x_out_f - x_in_f
if self.stf_score_fn == "relative_l2":
    delta = torch.linalg.vector_norm(diff, dim=-1, ord=2, keepdim=True)
    delta = delta / (torch.linalg.vector_norm(x_in_f, dim=-1, ord=2, keepdim=True) + 1e-6)
elif self.stf_score_fn == "cosine":
    cos = F.cosine_similarity(x_in_f, x_out_f, dim=-1, eps=1e-6).unsqueeze(-1)
    delta = (1.0 - cos) * 0.5
elif self.stf_score_fn == "direction":
    delta = F.cosine_similarity(diff, x_in_f, dim=-1, eps=1e-6).abs().unsqueeze(-1)
else:  # "l2" default
    delta = torch.linalg.vector_norm(diff, dim=-1, ord=2, keepdim=True)
```

**Important:** The `direction` formula inverts meaning (high = active). This is already consistent since all paths feed into `ema_score >= threshold` — just note that the threshold value needs to be calibrated differently for `direction` vs `l2`. Add a note in the log line.

This change applies to **the current working branch `claude/token-freezing-analysis-HS42X`** first, then cherry-pick to individual STF branches.

---

## Part 3: Compute Reduction (X + Y - A)

### The goal: find A within baseline X that becomes redundant with STF

**Candidate A1: Warmup steps (most concrete)**
- Where: `train_gpt.py`, `WARMUP_STEPS` (default 20)
- What it does: runs 20 full fwd+bwd+optimizer passes then resets all weights/optimizer state — purely to prime compiled CUDA kernels
- With STF branches: `compile_model=False` by default, so warmup is priming uncompiled paths. With no `torch.compile`, the warmup value drops further.
- **Recommendation: `WARMUP_STEPS=0` for STF branches** (skip entirely, or `WARMUP_STEPS=5` if stability is a concern)
- Savings: `warmup_steps × grad_accum_steps = 20 × 8 = 160` forward+backward passes saved

**Candidate A2: Muon backend steps**
- Where: `MUON_BACKEND_STEPS` (default 5 Newton-Schulz iterations)
- What it does: 5 matrix-multiply iterations per matrix param per step for orthogonal update
- With STF: fewer effective "active" tokens per step → each step carries less gradient information → high-precision orthogonalization is overkill
- **Recommendation: `MUON_BACKEND_STEPS=3`**
- Savings: ~40% of Muon optimizer cost. Muon runs on all 2D block params (most params).

**Candidate A3: STF Telemetry (`STF_TELEMETRY=0` after smoke test)**
- Where: `_record_stf_stats()` called per layer per step (9 buffer updates × num_layers)
- Once score function is calibrated via 500-step smoke test, disable for longer runs
- Toggle: already supported via `STF_TELEMETRY=0` env var
- Savings: small (buffer ops), but reduces memory pressure

**Candidate A4: Validation frequency**
- Default `VAL_LOSS_EVERY=1000` → for smoke/mid tests use `VAL_LOSS_EVERY=50` or `200`
- This is not a training compute saving but reduces wall-clock time during short runs

**NOT recommended to remove:**
- Skip connections (`skip_weights`): core architectural feature, removing changes training dynamics
- Grad clipping: already off by default (`GRAD_CLIP_NORM=0.0`)

### Net impact estimate:
- A1 (warmup=0): ~0.1% of total compute (20 warmup / ~20000 training steps)
- A2 (backend_steps=3): ~1-3% of total step compute (Muon is O(D²) but less than attention)
- Combined: modest but free, no quality loss expected

---

## Part 4: Progressive Test Protocol

Rationale: budget-conscious, validate each formula before going longer.

### Stage 1 — Smoke test (500 iterations, ~1-2 min on single GPU)
```bash
export ITERATIONS=500
export VAL_LOSS_EVERY=50
export STF_TELEMETRY=1        # keep on to verify formula is producing scores
export WARMUP_STEPS=0
export MUON_BACKEND_STEPS=3
```
Goal: confirm `stf_stats` shows meaningful score distribution, `active_mean` in [0.35, 0.65] range. If `active_mean > 0.95`, threshold needs lowering. If `active_mean < 0.10`, threshold too high.

Run for each of the 4 score functions (`l2`, `relative_l2`, `cosine`, `direction`) on `stf-minimal` mode.

### Stage 2 — Mid validation (2k or 5k iterations)
Decision rule from Stage 1: pick the 2 best-looking score functions (based on `active_mean` stability and `val_bpb` trajectory).

```bash
export ITERATIONS=2000        # or 5000 if Stage 1 looks promising
export VAL_LOSS_EVERY=200
export STF_TELEMETRY=0        # disable after smoke test
export WARMUP_STEPS=0
export MUON_BACKEND_STEPS=3
```
Compare `val_bpb` curves across score functions at same step count.

### Stage 3 — Full short run (10k iterations)
Only for the winner from Stage 2. Compare to baseline at 10k steps.

```bash
export ITERATIONS=10000
export VAL_LOSS_EVERY=500
export STF_TELEMETRY=0
export WARMUP_STEPS=0
export MUON_BACKEND_STEPS=3
```

---

## Critical Files to Modify

1. **`train_gpt.py`** on working branch `claude/token-freezing-analysis-HS42X`:
   - `Hyperparameters` class: add `stf_score_fn = os.environ.get("STF_SCORE_FN", "l2")`
   - `GPT.__init__` signature: add `stf_score_fn: str`
   - `GPT.__init__` body: store `self.stf_score_fn = stf_score_fn`
   - `GPT.forward` → `apply_stf()` inner function: replace single `delta` line with 4-branch conditional
   - `main()` → GPT instantiation: pass `stf_score_fn=args.stf_score_fn`
   - `main()` → startup log: add `stf_score_fn:{args.stf_score_fn}` to the stf config log line

2. **`My_approch/stf_analysis.md`** (new file): written summary of findings for future sessions

---

## Verification

- Run `STF_SCORE_FN=l2 ITERATIONS=200 torchrun ... train_gpt.py` — should produce same `stf_stats` as current branches (sanity check no regression)
- Run `STF_SCORE_FN=relative_l2 ITERATIONS=200 ...` — `score_mean` should be much smaller (< 0.1 typical) vs l2 score (~3-10 range)
- Run `STF_SCORE_FN=cosine ITERATIONS=200 ...` — score should be in [0, 0.5] range (cosine distances near 0 for smoothly-evolving tokens)
- Check `active_by_layer` shows per-layer variation (some layers freeze more than others)
- Confirm `val_bpb` is finite and decreasing over first 200 steps

After smoke test, cherry-pick the `STF_SCORE_FN` addition to each STF branch via `git cherry-pick` or manual patch.