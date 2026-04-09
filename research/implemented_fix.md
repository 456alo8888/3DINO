# 3DINO SOOP Regression Stability Fix - Implemented

## Date
2026-03-19

## Goal
Fix exploding-loss behavior in SOOP regression training (`dinov2/eval/linear3d_soop.py`) documented in `research/problem_training.md`.

## Files Updated
- `dinov2/eval/setup.py`
- `dinov2/eval/linear3d_soop.py`
- `research/run_soop_outcome_experiments.sh`
- `research/experiment_manifest.md`
- `research/problem_training.md`
- `research/plan_fix.md`

## What Was Implemented

### 1) Strict pretrained guard
- Added `--strict-pretrained` in eval setup parser.
- In strict mode, missing checkpoint now raises error and stops execution (no silent random-init fallback).

### 2) Stable optimizer controls for regression head
- Added optimizer CLI controls in `linear3d_soop.py`:
  - `--optimizer {auto,sgd,adamw}`
  - `--sgd-momentum`
  - `--adamw-beta1`, `--adamw-beta2`, `--adamw-eps`
- `auto` resolves to:
  - `adamw` for regression,
  - `sgd` for classification modes.

### 3) Feature normalization and gradient clipping
- Added `--normalize-features` to L2-normalize extracted image feature vectors before head input.
- Added `--grad-clip-norm` and clipping after backward before optimizer step.

### 4) Head-only safety and diagnostics
- Explicitly freeze backbone parameters (`requires_grad=False`) in SOOP linear eval path.
- Runtime assertion ensures backbone trainable parameter count is zero.
- Added diagnostics logging with `--diagnostics-period`:
  - `grad_norm`,
  - `feature_absmax`,
  - `feature_norm_mean`,
  - `pred_absmax`.

### 5) Result metadata and W&B config enrichment
- Added `training_config` block to `results_eval_linear_soop.json` with optimizer and stability settings.
- Extended W&B init config to include optimizer/normalization/clipping/strict flags.

### 6) Runner alignment
- Updated `research/run_soop_outcome_experiments.sh`:
  - Stability env controls:
    - `OPTIMIZER` (default `auto`)
    - `NORMALIZE_FEATURES` (default `1`)
    - `GRAD_CLIP_NORM` (default `1.0`)
    - `STRICT_PRETRAINED` (default `1`)
  - Added optional baseline comparison block:
    - `RUN_BASELINE_COMPARE=1` runs old-style baseline profile (`sgd`, no norm, no clip).

## Automated Verification Executed

### A) CLI and parser checks
- `python dinov2/eval/linear3d_soop.py --help`
- Verified new flags are present:
  - `--strict-pretrained`, `--optimizer`, `--normalize-features`, `--grad-clip-norm`, `--diagnostics-period`.

### B) Strict-pretrained fail-fast
- Ran with invalid checkpoint + `--strict-pretrained`.
- Observed expected fail-fast with `FileNotFoundError` and non-zero exit code.

### C) Runner syntax
- `bash -n research/run_soop_outcome_experiments.sh` passed.

### D) Stability short-run (fix profile)
- Command profile: `optimizer=adamw`, `normalize-features=on`, `grad-clip-norm=1.0`, `epoch-length=20`.
- Observed diagnostics:
  - `iter=0 loss=2.283813`, `grad_norm=3.273497`
  - `iter=10 loss=3.013942`, `grad_norm=3.869520`
- No loss explosion to extreme scale in early iterations.

### E) Backward compatibility (SGD path)
- Quick run with `--optimizer sgd` completed successfully.
- Confirms legacy optimizer path still executable.

### F) Result JSON schema check
- Verified `/tmp/soop_fix_adamw_norm_clip/results_eval_linear_soop.json` contains:
  - `training_config` with stability keys,
  - full regression metric keys in val/test: `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`.

## Plan Checklist Updated
- Updated completed automated checkboxes in `research/plan_fix.md` for:
  - Phase 1 automated checks,
  - Phase 2 automated checks,
  - Phase 3 shell syntax check.
- Manual verification checkboxes remain unchecked pending user validation.

## Notes
- One attempted run on GPU 0 failed due to external VRAM pressure (`torch.OutOfMemoryError`); rerun on GPU 2 succeeded.
- `mape` can still be numerically very large when targets are near zero; this is expected behavior of percentage-based error and does not indicate loss explosion by itself.
