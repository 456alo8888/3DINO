# 3DINO SOOP Outcome - Implemented

## Date
2026-03-19

## Scope Completed
Implemented code for using 3DINO backbone in downstream stroke outcome experiments on SOOP splits, including image-only and image+tabular modes.

## Files Added
- `dinov2/data/soop_dataset.py`
  - New `SOOPOutcomeDataset` adapter on top of `datasets.SOOP_dataset.SOOPTraceTabularDataset`.
  - Exposes downstream keys: `image`, `label`, `label_mask`, optional `tabular`, optional `subject_id`.
- `dinov2/eval/linear3d_soop.py`
  - New SOOP-focused evaluation/training entry script.
  - Supports `--target-col`, `--task-type`, `--drop-missing-labels`, `--use-tabular`.
  - Supports image-only and multimodal image+tabular training.
  - Writes `results_eval_linear_soop.json` and checkpoints.
- `research/run_soop_outcome_experiments.sh`
  - End-to-end experiment runner for image-only, image+tabular, and NIHSS regression variants.
- `research/experiment_manifest.md`
  - Run manifest and expected artifacts.

## Files Updated
- `dinov2/data/loaders.py`
  - Added `SOOP` branch in `make_classification_dataset_3d`.
  - Added optional args: `target_col`, `include_tabular`, `drop_missing_labels`.
- `dinov2/data/transforms.py`
  - Added `dataset_name == 'SOOP'` transform branch.
- `dinov2/data/__init__.py`
  - Exported `SOOPOutcomeDataset`.
- `README.md`
  - Added SOOP stroke outcome evaluation command examples.
- `research/plan.md`
  - Updated automated verification checkboxes for completed checks.

## Environment / Dependencies
Installed into `hieupcvp` for this implementation:
- `monai`
- `torchio`
- `torchmetrics`
- (`fvcore` already present)

## Automated Verification Executed
1. Import checks:
- `from dinov2.data.loaders import make_classification_dataset_3d` -> OK

2. SOOP dataset instantiation check:
- Built train/valid/test from `/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold`
- Verified sample keys include `image`, `label`, `label_mask`, `tabular`

3. Script CLI check:
- `python dinov2/eval/linear3d_soop.py --help` -> OK

4. End-to-end smoke runs (with free GPUs)
- Image-only binary run -> OK
  - Output: `/tmp/soop_eval_out_image/results_eval_linear_soop.json`
  - Checkpoint: `/tmp/soop_eval_out_image/best_val_soop.pth`
- Image+tabular binary run -> OK
  - Output: `/tmp/soop_eval_out_tab/results_eval_linear_soop.json`
  - Checkpoint: `/tmp/soop_eval_out_tab/best_val_soop.pth`

5. Runner script static check:
- `bash -n research/run_soop_outcome_experiments.sh` -> OK

## How to Run

### 1) Activate environment (or use conda run)
```bash
conda activate hieupcvp
```

### 2) Run image-only binary outcome
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /path/to/teacher_checkpoint.pth \
  --output-dir /path/to/output/soop_gsrankin_image_only \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --cache-dir /tmp/soop_cache \
  --target-col gs_rankin_6isdeath \
  --task-type binary \
  --batch-size 8 \
  --epochs 10 \
  --epoch-length 125 \
  --eval-period-iterations 125
```

### 3) Run image+tabular binary outcome
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /path/to/teacher_checkpoint.pth \
  --output-dir /path/to/output/soop_gsrankin_image_tabular \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --cache-dir /tmp/soop_cache \
  --target-col gs_rankin_6isdeath \
  --task-type binary \
  --use-tabular \
  --batch-size 8 \
  --epochs 10 \
  --epoch-length 125 \
  --eval-period-iterations 125
```

### 4) Run NIHSS regression
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /path/to/teacher_checkpoint.pth \
  --output-dir /path/to/output/soop_nihss_regression \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --cache-dir /tmp/soop_cache \
  --target-col nihss \
  --task-type regression \
  --use-tabular \
  --batch-size 8 \
  --epochs 10 \
  --epoch-length 125 \
  --eval-period-iterations 125
```

### 5) Run bundled script
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
PRETRAINED_WEIGHTS=/path/to/teacher_checkpoint.pth bash research/run_soop_outcome_experiments.sh
```

## Notes
- If GPU 0 is occupied, set `CUDA_VISIBLE_DEVICES` to a free GPU before running.
- The command `humanlayer thoughts sync` is still unavailable in this machine (`command not found`).

---

## Addendum Implemented (Regression-first + Full Metrics + W&B)

### Date
2026-03-19

### Scope Completed
Implemented the addendum in `research/plan.md` to shift SOOP outcome runs to regression-first defaults, expand regression metric reporting, and add optional Weights & Biases tracking.

### Files Updated
- `dinov2/eval/linear3d_soop.py`
  - Default `--task-type` switched to `regression`.
  - Added regression-first guardrail for outcome targets (`nihss`, `gs_rankin_6isdeath`, alias), with explicit override flag `--allow-classification-outcome`.
  - Added full regression metrics for val/test: `mse`, `rmse`, `mae`, `mape`, `r2`, `loss`.
  - Extended output JSON with:
    - `selection_metric`
    - `train_metrics` (`loss_last`, `loss_mean`, `iterations`)
    - rich `val_metric.metrics` and `test_metric.metrics` dictionaries.
  - Added optional W&B integration:
    - `--use-wandb`, `--wandb-project`, `--wandb-entity`, `--wandb-run-name`, `--wandb-tags`, `--wandb-mode`.
    - Logs train loss/lr, periodic val metrics, and final test metrics.
- `research/run_soop_outcome_experiments.sh`
  - Converted gsrankin runs from binary to regression.
  - Added W&B env handling and export path:
    - `USE_WANDB`, `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`, `WANDB_TAGS`, `WANDB_RUN_PREFIX`.
  - Added per-run W&B run names.
- `README.md`
  - Updated SOOP outcome command example to regression.
  - Added note for full regression metrics.
  - Added optional W&B run example.
- `research/experiment_manifest.md`
  - Updated run definitions to regression-first.
  - Added expected regression metric bundle and W&B env overrides.
- `requirements.txt`
  - Added `wandb` dependency.

### Updated How to Run (Regression-first)

#### A) Image-only gsrankin regression
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /path/to/teacher_checkpoint.pth \
  --output-dir /path/to/output/soop_gsrankin_image_only \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --cache-dir /tmp/soop_cache \
  --target-col gs_rankin_6isdeath \
  --task-type regression \
  --batch-size 8 \
  --epochs 10 \
  --epoch-length 125 \
  --eval-period-iterations 125
```

#### B) Image+tabular gsrankin regression
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /path/to/teacher_checkpoint.pth \
  --output-dir /path/to/output/soop_gsrankin_image_tabular \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --cache-dir /tmp/soop_cache \
  --target-col gs_rankin_6isdeath \
  --task-type regression \
  --use-tabular \
  --batch-size 8 \
  --epochs 10 \
  --epoch-length 125 \
  --eval-period-iterations 125
```

#### C) Image+tabular NIHSS regression
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /path/to/teacher_checkpoint.pth \
  --output-dir /path/to/output/soop_nihss_regression \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --cache-dir /tmp/soop_cache \
  --target-col nihss \
  --task-type regression \
  --use-tabular \
  --batch-size 8 \
  --epochs 10 \
  --epoch-length 125 \
  --eval-period-iterations 125
```

#### D) Bundled runner (with optional W&B)
```bash
cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO

# local only
PRETRAINED_WEIGHTS=/path/to/teacher_checkpoint.pth USE_WANDB=0 \
  bash research/run_soop_outcome_experiments.sh

# with wandb
PRETRAINED_WEIGHTS=/path/to/teacher_checkpoint.pth \
USE_WANDB=1 \
WANDB_API_KEY=your_key \
WANDB_PROJECT=3dino-soop-outcome \
WANDB_ENTITY=your_entity \
WANDB_TAGS=soop,regression \
bash research/run_soop_outcome_experiments.sh
```
