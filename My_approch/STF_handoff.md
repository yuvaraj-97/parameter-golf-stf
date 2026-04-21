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

---

## 9) Handoff update, 2026-04-21, unified HTML report work

Current user ask that was handled:

> `index.html` should not be a link hub. The STF summaries, old branch-comparison data, missing/gap rows, and latest Vast iterations should be brought together into one complete page.

### Current worktree state

Known modified files:

- `index.html`
- `scripts/summarize_vast_formula_logs.py`

Known untracked path that already existed during this handoff:

- `analysis/`

Do not assume `analysis/` belongs to the HTML unification work. Inspect it before touching it.

### What changed

`index.html` was regenerated as the unified report itself. It is no longer a "Report Hub" that mainly links out to other HTML pages.

The root page now includes:

- Unified hero summary from the latest Vast logs.
- Internal section navigation only: branch tree, validation ladder, legacy coverage, reality table.
- Branch tree and actual git ref graph directly in the root HTML.
- Validation ladder for:
  - fixed `codex/stf-mlp-skip-runner` relative_l2 active 0.75 at 2k, 5k, 10k
  - adaptive `codex/stf-adaptive-mlp-budget` relative_l2 at 500, 2k, 10k
- Embedded "Legacy Iteration Coverage" section from `My_approch/analysis/*_branch_compare_data.json`.
- Explicit old-data gaps as missing rows, rather than hiding branches that did not have a completed run.
- Existing visual sections from the generated report: compare picker, freeze mini movies, layer diagrams, outcome view, speed reality, and reality-check table.

`scripts/summarize_vast_formula_logs.py` was updated so the unified root page is reproducible. New/changed generator behavior:

- Adds `load_legacy_analysis_reports()`.
- Adds `as_float()`.
- Adds `render_legacy_analysis_coverage()`.
- Adds stable IDs for internal anchor navigation:
  - `#branch-tree`
  - `#validation-ladder`
  - `#legacy-coverage`
  - `#reality-check`
- Removes old root-page link-hub behavior from the generated report header.
- Adds `.status.main` styling.

### Regeneration command used

From repo root:

```bash
python3 scripts/summarize_vast_formula_logs.py vast*.log vast/experiments/**/console.log --html index.html
```

This command successfully wrote:

```text
Wrote HTML report: index.html
```

### Verification already done

Checked that the generated root page contains:

- `Unified STF report, Vast logs + imported branch-comparison data`
- `Legacy Iteration Coverage`
- `#branch-tree`
- `#validation-ladder`
- `#legacy-coverage`
- `#reality-check`

Checked that old link-hub labels are gone from `index.html`:

- `Report hub`
- `Imported analysis bundle`
- `Standalone branch tree`

Ran a no-cache syntax compile check for the generator:

```bash
python3 -c "compile(open('scripts/summarize_vast_formula_logs.py', encoding='utf-8').read(), 'scripts/summarize_vast_formula_logs.py', 'exec')"
```

It passed with no output.

Note: `python3 -m py_compile scripts/summarize_vast_formula_logs.py` failed in the sandbox because macOS Python tried to write bytecode under `/Users/yuvraj/Library/Caches/com.apple.python/...`, which is outside the writable sandbox. That was a sandbox/cache-location issue, not a syntax issue.

### Important data points now visible in `index.html`

Latest parsed Vast summary:

- Best current run: `codex/stf-mlp-skip-runner / relative_l2`, 10k, final BPB about `1.2806`.
- Adaptive budget run: `codex/stf-adaptive-mlp-budget / relative_l2`, 10k, final BPB about `1.2846`.
- Fixed MLP skip actual skip ratio is about `20.8%`.
- Adaptive MLP budget actual skip ratio is about `29.2%`.
- Some 500-step budget runs show higher actual skip, so be careful interpreting the top-line max actual skip metric; the validation ladder is the better quality signal.

Legacy embedded coverage:

- 4 legacy data bundles found.
- 21 completed old rows.
- 11 explicit old gaps/missing rows.
- 8 branch definitions.

### Suggested next-session actions

1. Open `file:///Users/yuvraj/Documents/GitHub/parameter-golf-stf/index.html` and visually skim the unified report.
2. If the top-line max actual skip feels misleading because it is pulled from a lower-quality 500-step sweep, adjust the generator to show "best 10k actual skip" or "validated actual skip" beside/instead of max actual skip.
3. Decide whether to also regenerate `vast_formula_summary.html` with the same unified generator output, or intentionally leave it as the older standalone current-report artifact.
4. Review and optionally stage/commit only:
   - `index.html`
   - `scripts/summarize_vast_formula_logs.py`
   - `My_approch/STF_handoff.md`

Avoid staging `analysis/` unless the user explicitly confirms what it is.

### Follow-up update: all branch/run artifacts included

After the first handoff entry, the user asked for confirmation that every run across branches was represented in the HTML. The answer was "not strictly yet", because the page had completed STF/Vast summaries and legacy analysis JSON coverage, but not every raw run artifact.

The generator and `index.html` were then updated again with an "All Discovered Run Logs" section, then corrected after the user clarified that `records/**` entries are not theirs and should not be included:

- Section anchor: `#all-run-inventory`
- Header nav now includes `All runs`.
- The section scans:
  - top-level `vast*.log`
  - `vast/experiments/**/console.log`
  - `vast/experiments/**/train.log`
- Parsed STF/Vast logs are listed per branch/formula variant, so multi-variant files like `vast_pod_a_500.log` and `vast_pod_b_500.log` do not collapse into a misleading single row.
- `records/**` artifacts are intentionally excluded from discovery and from the generated HTML.
- Failed/partial/seen artifacts, including query-sparse attempts, are listed instead of being hidden.
- The branch comparison area is now a two-branch explorer with Branch A / Branch B selects, a validation BPB curve graph for the selected branches, branch descriptions, best-variant stats, a best-to-best delta summary, and a per-variant comparison table.

Current generated `index.html` reports:

- The all-run ledger is limited to top-level Vast logs plus `vast/experiments/**/{console,train}.log`.
- No `records/**` rows should appear in the all-run ledger.
- `codex/stf-query-sparse-attn` failed/seen attempts appear in the all-run ledger.

Verification after this update:

```bash
python3 scripts/summarize_vast_formula_logs.py vast*.log vast/experiments/**/console.log --html index.html
python3 -c "compile(open('scripts/summarize_vast_formula_logs.py', encoding='utf-8').read(), 'scripts/summarize_vast_formula_logs.py', 'exec')"
rg -n "href=\"#all-run-inventory|All Discovered Run Logs|run groups discovered|branchASelect|branchCompareChart|codex/stf-query-sparse-attn" index.html
rg -n "records/" index.html scripts/summarize_vast_formula_logs.py
```

The final `records/` check should return no generated-report/generator matches.

### Follow-up update: layout/readability fixes

After reviewing the rendered HTML screenshots, the report was adjusted again:

- Table CSS no longer forces a fixed 1040px minimum width, and cells can wrap long values instead of forcing horizontal scroll while nearby screen space is unused.
- The speed reality card now uses one full-width column, so the table is not trapped in the right half of a two-column layout.
- The hero metrics now include `completed 10k variants`; current generated value is `2`.
- The non-technical outcome chart now uses a wider label lane, fewer rows, no overlapping row text, and the unlabeled white overall-score dot was removed.
- "Layer Freezing Diagrams" was collapsed into an optional `Per-layer final snapshots` section because the expanded diagrams repeated the same visual idea as the Freeze Mini Movies. The distinction is now that mini movies show training-time motion, while snapshots are final per-layer telemetry.
- The all-run inventory was later changed from a 10-column table to responsive run cards. This avoids both horizontal scrolling and the letter-by-letter column wrapping seen in the rendered screenshot.
- The header nav was duplicated into a fixed bottom floating nav so navigation is always available. It includes a `Refresh data` button.

Verification after this update:

```bash
python3 -c "compile(open('scripts/summarize_vast_formula_logs.py', encoding='utf-8').read(), 'scripts/summarize_vast_formula_logs.py', 'exec')"
python3 scripts/summarize_vast_formula_logs.py vast*.log vast/experiments/**/console.log --html index.html
node --check /dev/stdin < <(python3 - <<'PY'
from pathlib import Path
text = Path('index.html').read_text()
start = text.index('<script>') + len('<script>')
end = text.index('</script>', start)
print(text[start:end])
PY
)
rg -n "completed 10k variants|Per-layer final snapshots|total-dot|Layer Freezing Diagrams" index.html scripts/summarize_vast_formula_logs.py
```
