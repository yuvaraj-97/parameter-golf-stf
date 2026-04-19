# STF Next-Step Plan And Handoff Update

## Summary
The current STF path is promising as an idea, but **not promising enough in its current implementation/config** to scale directly. On 80 shards, baseline beats all STF branches at 5k.

Current 80-shard 5k result:

| Branch | Roundtrip BPB | Raw BPB | Step Avg | Verdict |
|---|---:|---:|---:|---|
| `baseline-repro` | `1.25187638` | `1.2475` | `389.39ms` | Current winner |
| `stf-learned-gate` | `1.26287554` | `1.2585` | `612.29ms` | Worse quality, slower |
| `stf-recurrence` | `1.26902905` | `1.2649` | `614.64ms` | Worse quality, slower |
| `stf-soft-freeze` | `1.27656162` | `1.2727` | `613.90ms` | Worse quality, slower |

The public naive baseline target is `1.2244`. Baseline-repro at 80-shard 5k is `1.2519`, so we are about `0.0275 BPB` away. The baseline curve is still improving strongly at 5k, so **10k baseline is justified**. Direct 20k is premature until we see 10k.

## Key Decisions
- Run `baseline-repro 10000` next with 80 shards.
- Do not run current STF branches at 10k/20k unchanged.
- Do not use `COMPILE_MODEL=1` yet for STF branches.
- Use compile mode only after a branch is quality-competitive; speed optimization before quality is premature.
- Keep STF research active, but change the scoring formula before scaling again.

## STF Formula Direction
Current STF uses:

```python
delta = ||x_out - x_in||
```

This seems too crude under 80-shard training. It may freeze/blend tokens based on hidden-state movement, not based on whether the token still needs useful computation.

Next STF experiment should replace or augment the score with one of these, in this order:

1. **Normalized residual change**
   ```text
   ||x_out - x_in|| / (||x_in|| + eps)
   ```
   This is the safest first change because it avoids high-norm tokens dominating the freeze decision.

2. **Cosine direction change**
   ```text
   1 - cosine_similarity(x_out, x_in)
   ```
   This captures semantic direction shift rather than raw magnitude.

3. **Attention/MLP residual split score**
   ```text
   score = alpha * ||attn_out|| + beta * ||mlp_out||
   ```
   This requires deeper code changes but may tell us whether tokens are changing because of attention or FFN.

4. **Loss-proxy/token confidence score**
   ```text
   use token logit entropy or margin to keep uncertain tokens active
   ```
   This is potentially stronger but more expensive and should not be first.

First implementation target: `stf-learned-gate` with normalized residual change. Learned gate already has the softest routing, so it is the best branch to test a better score formula.

## Baseline And Scaling Plan
Run:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
echo "train_shards=$(ls ./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin | wc -l)"
bash scripts/run_train_and_save.sh baseline-repro 10000 1
```

Decision after 10k baseline:

| 10k baseline roundtrip BPB | Next step |
|---:|---|
| `<= 1.2244` | Run `baseline-repro 20000`; start compression cleanup |
| `1.2244` to `1.235` | Run `baseline-repro 20000`; still plausible |
| `> 1.235` | Stop scaling baseline; add architecture/compression tricks first |

Current baseline step time worsened from `234.33ms` to `389.39ms` when moving from 1 shard to 80 shards, likely due to dataset/I/O regime and fuller training stream. STF worsened less percentage-wise, but absolute STF speed is still much slower than baseline. Quality still dominates the decision.

## PR Tricks To Cherry-Pick Later
Do not add these before the 10k baseline. After the 10k result, consider one controlled add-on at a time:

- **Longer context / sliding eval**: leaderboard shows 2048/4096 context and sliding evaluation helped.
- **QK gain tuning**: appears in multiple top records and is low-risk to test.
- **Parallel residuals**: common in strong submissions, but more invasive.
- **Depth recurrence**: aligned with our STF theme, but current STF recurrence is not the same as proven architectural recurrence.
- **Compression upgrades**: int6/mixed precision, GPTQ-lite, zstd. Artifact margin is only about `150k`, so compression must be addressed before serious final submission.

## Compile Mode Plan
Do not run compile mode now for the main path.

Compile-mode sequence only after quality is competitive:

```bash
COMPILE_MODEL=1 bash scripts/run_train_and_save.sh <candidate-branch> 200 1
COMPILE_MODEL=1 bash scripts/run_train_and_save.sh <candidate-branch> 2000 1
```

Only if both pass and BPB is unchanged should compile mode be used for 5k/10k/20k.

For baseline, compile mode may already work better than STF, but it is not the next bottleneck. The next bottleneck is whether baseline reaches `1.2244` at 10k/20k and whether STF can beat it after formula changes.

## Handoff Document Update
When execution is allowed, update:

```text
My_approch/STF_handoff.md
```

Replace the stale branch-order/runbook content with:
- Current date and repo state.
- 80-shard 5k ranking table.
- Explicit conclusion: baseline wins; current STF branches should not scale unchanged.
- Next run: `baseline-repro 10000`.
- Decision gate for 10k -> 20k.
- STF formula-change backlog.
- Compile-mode policy.
- Copy/paste commands for 80-shard baseline 10k and analysis regeneration.
- Warning that 1-shard runs are smoke/debug only and should not drive challenge decisions.

Acceptance criteria for the handoff:
- A new session can answer “what should I run next?” without reading old chat.
- It clearly distinguishes 1-shard vs 80-shard results.
- It says not to scale current STF branches unchanged.
- It includes the exact next command and the exact decision gate after 10k.
