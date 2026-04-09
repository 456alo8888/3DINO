# SOOP Raw TRACE Stroke Outcome Experiment Manifest

## Environment
- Conda env: `hieupcvp`
- Repository root: `baseline_encoder/3DINO`
- Eval script: `dinov2/eval/linear3d_soop.py`

## Shared Inputs
- Config file: `dinov2/configs/train/vit3d_highres.yaml`
- Pretrained teacher checkpoint: `<set PRETRAINED_WEIGHTS>`
- RAW split directory: `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace`
- Seed: `0`

## Runs

### Run A: RAW image-only gsrankin regression
- Output dir: `outputs/soop_raw_gsrankin_image_only`
- Target: `gs_rankin_6isdeath`
- Task type: `regression`
- Tabular: `disabled`

### Run B: RAW image+tabular gsrankin regression
- Output dir: `outputs/soop_raw_gsrankin_image_tabular`
- Target: `gs_rankin_6isdeath`
- Task type: `regression`
- Tabular: `enabled`

### Run C: RAW image+tabular NIHSS regression
- Output dir: `outputs/soop_raw_nihss_regression`
- Target: `nihss`
- Task type: `regression`
- Tabular: `enabled`

## Expected Artifacts Per Run
- `results_eval_linear_soop.json`
- `best_val_soop.pth`
- `checkpoint_iter_*.pth` (if checkpoint period reached)
- Regression metrics in JSON:
  - `mse`
  - `rmse`
  - `mae`
  - `mape`
  - `r2`
  - `loss`

## W&B Logging
- Enable with `USE_WANDB=1` and `WANDB_API_KEY`.
- Key W&B params:
  - `WANDB_PROJECT`
  - `WANDB_ENTITY`
  - `WANDB_MODE`
  - `WANDB_TAGS`
  - `WANDB_RUN_PREFIX`
- Metrics logged from `linear3d_soop.py` include train diagnostics, val metrics, and final test metrics.

## Reproducible Command
Use:
- `research/run_soop_outcome_experiments_raw.sh`

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
