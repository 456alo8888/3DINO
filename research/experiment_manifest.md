# SOOP Stroke Outcome Experiment Manifest

## Environment
- Conda env: `hieupcvp`
- Repository root: `baseline_encoder/3DINO`
- Eval script: `dinov2/eval/linear3d_soop.py`

## Shared Inputs
- Config file: `dinov2/configs/train/vit3d_highres.yaml`
- Pretrained teacher checkpoint: `<set PRETRAINED_WEIGHTS>`
- SOOP split directory: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold`
- Seed: `0`

## Runs

### Run A: Image-only gsrankin regression
- Output dir: `outputs/soop_gsrankin_image_only`
- Target: `gs_rankin_6isdeath`
- Task type: `regression`
- Tabular: `disabled`

### Run B: Image+tabular gsrankin regression
- Output dir: `outputs/soop_gsrankin_image_tabular`
- Target: `gs_rankin_6isdeath`
- Task type: `regression`
- Tabular: `enabled`

### Run C: Image+tabular NIHSS regression
- Output dir: `outputs/soop_nihss_regression`
- Target: `nihss`
- Task type: `regression`
- Tabular: `enabled`

## Expected Artifacts Per Run
- `results_eval_linear_soop.json`
- `best_val_soop.pth`
- `checkpoint_iter_*.pth` (if checkpoint period reached)
- Regression metrics in JSON: `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`

## Validation Matrix (Exploding-Loss Fix)

### Stability Profile (recommended)
- `optimizer=adamw` (hoặc `optimizer=auto` với regression)
- `normalize-features=on`
- `grad-clip-norm=1.0`
- `strict-pretrained=on`

### Matrix
1. `gs_rankin_6isdeath` + image-only + stability profile
2. `gs_rankin_6isdeath` + image+tabular + stability profile
3. `nihss` + image+tabular + stability profile

### Optional Baseline Ablation
- `gs_rankin_6isdeath` + image-only + `optimizer=sgd`, `normalize-features=off`, `grad-clip-norm=0.0`

## Reproducible Command
Use:
- `research/run_soop_outcome_experiments.sh`

Override by environment variables:
- `PRETRAINED_WEIGHTS`
- `FOLD_DIR`
- `CACHE_DIR`
- `CONFIG_FILE`
- `USE_WANDB`
- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY`
- `WANDB_MODE`
- `WANDB_TAGS`
- `WANDB_RUN_PREFIX`
