# 3DINO Stroke Outcome (SOOP) Implementation Plan

## Overview

Implement a reproducible 3DINO downstream experiment pipeline for stroke treatment-outcome prediction using the SOOP dataset interface in `datasets/SOOP_dataset.py`, with both image-only and image+tabular experiment tracks.

## Current State Analysis

The codebase already has:
- 3D classification downstream pipeline in `dinov2/eval/linear3d.py` (frozen backbone + linear heads).
- Classification dataset/transform switch points in `dinov2/data/loaders.py` and `dinov2/data/transforms.py`.
- SOOP dataset contract in `datasets/SOOP_dataset.py` returning:
  - `image` (3D channel-first tensor),
  - `tabular`,
  - `target` (multi-target vector),
  - `label_mask`.

Missing pieces for stroke outcome experiments:
- 3DINO does not currently have a SOOP dataset branch in `make_classification_dataset_3d`.
- `linear3d.py` expects scalar `label` in each sample, not SOOP `target` + `label_mask`.
- No experiment runner for outcome target selection (e.g., `gs_rankin_6isdeath` / `nihss`) and no multimodal (image+tabular) downstream head path.

## Desired End State

A complete experiment flow exists and is runnable in `hieupcvp` environment:
1. A SOOP-compatible downstream dataset path is available under 3DINO data module.
2. A stroke outcome eval script can train/evaluate image-only linear heads with explicit target selection and label-mask filtering.
3. A multimodal experiment variant can consume image features + SOOP tabular vector.
4. Standardized outputs are produced (metrics JSON + prediction CSV/JSON + checkpoint directory).

### Key Discoveries
- `dinov2/eval/linear3d.py` trains linear heads from `ModelWithIntermediateLayers` output and expects batch dict keys `image` and `label`.
- `dinov2/data/loaders.py` centralizes dataset-name dispatch and split handling for classification tasks.
- `datasets/SOOP_dataset.py` already supports missing-label handling (`allow_missing_labels`) and keeps per-target label masks.
- `dinov2/data/transforms.py` is currently dataset-name switched (`ICBM`, `COVID-CT-MD`) and should host SOOP transform branch for consistency.

## What We're NOT Doing

- No changes to 3DINO SSL pretraining (`dinov2/train/train3d.py`, SSL losses, teacher-student architecture).
- No changes to segmentation pipeline (`dinov2/eval/segmentation3d.py` and segmentation heads).
- No hyperparameter sweep framework beyond the existing linear-head grid approach.
- No model architecture redesign of backbone ViT.

## Implementation Approach

Follow the existing 3DINO downstream classification pattern and add SOOP integration as a parallel dataset/task path:
- Keep backbone loading and feature extraction unchanged.
- Add a dedicated SOOP dataset adapter and new eval entry script for outcome-aware batching (target column + mask).
- Add optional multimodal head path (concat image feature + tabular vector) in the same eval workflow.
- Preserve output/report format style from `linear3d.py`.

## Phase 1: SOOP Dataset Bridge in 3DINO

### Overview
Add data-layer plumbing so 3DINO can build train/val/test datasets from SOOP split CSVs via existing SOOP dataset class.

### Changes Required

#### 1) New SOOP adapter module
**File**: `baseline_encoder/3DINO/dinov2/data/soop_dataset.py` (new)
**Changes**:
- Import `SOOPTraceTabularDataset` from `datasets.SOOP_dataset`.
- Add adapter class/function to map SOOP sample to 3DINO downstream sample format.
- Support target selection by index/name and label-mask filtering.

```python
class SOOPOutcomeDataset(torch.utils.data.Dataset):
    def __init__(self, split_csv, target_col="gs_rankin_6isdeath", transform=None, include_tabular=False):
        self.base = SOOPTraceTabularDataset(
            split_csv=split_csv,
            target_cols=("nihss", "gs_rankin_6isdeath"),
            transform=transform,
            allow_missing_labels=True,
        )
        self.target_index = self.base.target_cols.index(target_col)
        self.include_tabular = include_tabular

    def __getitem__(self, idx):
        s = self.base[idx]
        label_mask = s["label_mask"][self.target_index]
        out = {"image": s["image"], "label": s["target"][self.target_index], "label_mask": label_mask}
        if self.include_tabular:
            out["tabular"] = s["tabular"]
        return out
```

#### 2) Register SOOP dataset in classification loader
**File**: `baseline_encoder/3DINO/dinov2/data/loaders.py`
**Changes**:
- Add dataset name branch `SOOP` in `make_classification_dataset_3d`.
- Build train/valid/test datasets from `train.csv`, `valid.csv`, `test.csv` under provided fold dir.
- Return class/task metadata for selected objective.

```python
elif dataset_name == "SOOP":
    train_csv = os.path.join(base_directory, "train.csv")
    valid_csv = os.path.join(base_directory, "valid.csv")
    test_csv = os.path.join(base_directory, "test.csv")
    train_dataset = SOOPOutcomeDataset(train_csv, target_col=target_col, transform=train_transforms)
    val_dataset = SOOPOutcomeDataset(valid_csv, target_col=target_col, transform=val_transforms)
    test_dataset = SOOPOutcomeDataset(test_csv, target_col=target_col, transform=val_transforms)
```

#### 3) Add SOOP transform branch
**File**: `baseline_encoder/3DINO/dinov2/data/transforms.py`
**Changes**:
- Add `dataset_name == 'SOOP'` branch in `make_classification_transform_3d`.
- Reuse existing MONAI dictionary transforms aligned with stroke MRI input assumptions.

### Success Criteria

#### Automated Verification:
- [x] SOOP dataset import path resolves in 3DINO process:
  - `conda run -n hieupcvp python -c "from dinov2.data.loaders import make_classification_dataset_3d; print('ok')"`
- [x] SOOP train/valid/test datasets can be instantiated:
  - `conda run -n hieupcvp python -c "from dinov2.data.loaders import make_classification_dataset_3d; ..."`
- [x] One batch from dataloader is readable and includes expected keys:
  - `image`, `label`, `label_mask` (and `tabular` if enabled).

#### Manual Verification:
- [ ] Random samples correspond to expected subjects in split CSVs.
- [ ] Selected outcome target (`gs_rankin_6isdeath` or `nihss`) maps correctly from SOOP target vector.
- [ ] Missing-label rows are handled as intended for training/eval.

**Implementation Note**: After completing this phase and all automated verification passes, pause for manual confirmation before moving to Phase 2.

---

## Phase 2: Stroke Outcome Linear Eval Script (Image-Only)

### Overview
Create a dedicated eval script that mirrors `linear3d.py` but supports SOOP outcome target/mask semantics.

### Changes Required

#### 1) New eval entry script for SOOP outcome
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py` (new)
**Changes**:
- Start from `linear3d.py` structure.
- Add CLI args:
  - `--target-col` (`gs_rankin_6isdeath` or `nihss`),
  - `--task-type` (`binary`, `multiclass`, `regression`),
  - `--drop-missing-labels`.
- In training/eval loops, apply `label_mask` filtering before loss/metric update.

```python
valid = batch["label_mask"] > 0
images = batch["image"][valid]
labels = batch["label"][valid]
features = feature_model(images)
logits = linear_classifiers(features)
```

#### 2) Task-aware loss and metric wiring
**Files**:
- `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
- optionally `baseline_encoder/3DINO/dinov2/eval/metrics.py`
**Changes**:
- Binary/multiclass classification: cross-entropy + top-1 metrics.
- Regression path (if enabled): MSE/MAE logging in result file.

#### 3) README experiment command block
**File**: `baseline_encoder/3DINO/README.md`
**Changes**:
- Add SOOP outcome command examples with all required args.

### Success Criteria

#### Automated Verification:
- [x] Script argument parsing works:
  - `conda run -n hieupcvp python baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py --help`
- [x] Dry-run on small subset executes one train epoch without crash.
- [x] Result file generated at `output_dir/results_eval_linear_soop.json`.
- [x] Checkpoint file generated in `output_dir`.

#### Manual Verification:
- [ ] Logs show the effective count of labeled samples after mask filtering.
- [ ] Best-classifier selection is recorded consistently with validation metric.
- [ ] Test prediction records include subject identifier mapping if configured.

**Implementation Note**: After this phase passes automated checks, pause for manual confirmation before Phase 3.

---

## Phase 3: Multimodal Head Experiment (Image + Tabular)

### Overview
Add an optional multimodal branch for treatment-outcome prediction using concatenated image features and SOOP tabular vector.

### Changes Required

#### 1) Multimodal classifier head
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Add `--use-tabular` flag.
- Concatenate image representation with tabular tensor before linear head.

```python
img_feat = create_linear_input(tokens, use_n_blocks, use_avgpool)
if use_tabular:
    x = torch.cat([img_feat, tabular.float()], dim=-1)
else:
    x = img_feat
logits = head(x)
```

#### 2) Controlled experiment outputs
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Write separate results keys for:
  - image-only,
  - image+tabular.
- Keep JSON schema compatible with existing `linear3d.py` style.

### Success Criteria

#### Automated Verification:
- [x] `--use-tabular false` run succeeds and writes metrics.
- [x] `--use-tabular true` run succeeds and writes metrics.
- [x] Result JSON includes both experiment labels and comparable metrics.

#### Manual Verification:
- [ ] Tabular feature dimension in logs matches SOOP dataset cache dimension.
- [ ] Subject alignment between image and tabular vectors is correct.

**Implementation Note**: Pause for manual confirmation before Phase 4.

---

## Phase 4: Experiment Runner + Reproducibility Package

### Overview
Standardize experiment commands and produce reproducible execution protocol for stroke outcome studies.

### Changes Required

#### 1) Experiment shell runner
**File**: `baseline_encoder/3DINO/research/run_soop_outcome_experiments.sh` (new)
**Changes**:
- Add sequential commands for:
  - image-only `gs_rankin_6isdeath`,
  - image+tabular `gs_rankin_6isdeath`,
  - optional `nihss` target run.
- Use `conda run -n hieupcvp` for all commands.

#### 2) Result manifest and summary
**File**: `baseline_encoder/3DINO/research/experiment_manifest.md` (new)
**Changes**:
- Track output directories, seed, config file, pretrained weights path, split path, and target settings.

### Success Criteria

#### Automated Verification:
- [ ] Runner script executes end-to-end without syntax errors.
- [ ] Each configured run generates:
  - metrics JSON,
  - checkpoint,
  - prediction output artifact.
- [ ] Re-run with same seed reproduces identical split/sample counts.

#### Manual Verification:
- [ ] Final table clearly compares image-only vs image+tabular outcome performance.
- [ ] Output directory naming is unambiguous for future reruns.

**Implementation Note**: After this phase and automated verification pass, pause for final human review of experiment outputs.

---

## Testing Strategy

### Unit Tests
- Validate SOOP adapter target mapping and mask filtering behavior.
- Validate dataset branch dispatch for `dataset_name == 'SOOP'`.
- Validate multimodal head input dimension construction.

### Integration Tests
- End-to-end single-GPU smoke run from CLI with tiny epoch/iteration settings.
- End-to-end batch read + forward + metrics update for both image-only and multimodal modes.

### Manual Testing Steps
1. Run one tiny image-only experiment on `gs_rankin_6isdeath` and inspect logs/results.
2. Run one tiny image+tabular experiment and verify tabular dimension + output JSON.
3. Compare prediction outputs and sanity-check subject IDs against split CSV.

## Performance Considerations

- Reuse existing frozen-backbone + linear-head paradigm to keep training cost low.
- Keep dataloaders on MONAI/PyTorch path with moderate worker count and persistent workers where safe.
- Use small pilot settings first (short epochs) before full runs.

## Migration Notes

- No data migration is required.
- SOOP split CSV files in `code/datasets/fold` remain source-of-truth.
- New scripts/modules should remain additive to avoid breaking existing ICBM/COVID pipelines.

## References

- Research context: `baseline_encoder/3DINO/research/2026-03-19-3dino-soop-treatment-outcome.md`
- Existing linear eval: `baseline_encoder/3DINO/dinov2/eval/linear3d.py`
- Classification loader switch: `baseline_encoder/3DINO/dinov2/data/loaders.py`
- Classification transforms: `baseline_encoder/3DINO/dinov2/data/transforms.py`
- SOOP dataset interface: `datasets/SOOP_dataset.py`

---

## Addendum (2026-03-19): Regression-first outcome plan + full metrics + W&B

### Context for this addendum
Current `linear3d_soop.py` supports `binary|multiclass|regression`, but regression reporting is currently limited to `mse` and training-time logger prints `loss` only. The requested implementation direction is to shift outcome experiments to regression for `nihss` and `gs_rankin_6isdeath`, expand regression metric reporting, and add experiment logging to Weights & Biases.

### Updated Desired End State
1. SOOP outcome experiments run in regression mode for both targets (`nihss`, `gs_rankin_6isdeath`) as the primary path.
2. Validation/test outputs include full regression metric set: `mae`, `mape`, `r2`, `rmse`, `mse`, and `loss`.
3. Training loop logs metrics to console, JSON artifacts, and W&B (optional toggle-able integration).
4. Runner script supports `WANDB_API_KEY` export and standardized W&B run metadata.

### Scope Clarification for this addendum
- In scope: `dinov2/eval/linear3d_soop.py`, `research/run_soop_outcome_experiments.sh`, docs/manifests in `research/` and `README.md`.
- Out of scope: SSL pretraining, segmentation flow, non-SOOP datasets.

## Phase 5: Convert SOOP outcome flow to regression-first

### Overview
Refactor the SOOP downstream entry to prioritize regression behavior for `nihss` and `gs_rankin_6isdeath`, while preserving compatibility options only if still required by maintainers.

### Changes Required

#### 1) Regression-first CLI and guardrails
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Set default `--task-type regression` for SOOP outcome runs.
- Add explicit target-task validation to prevent accidental binary conversion for outcome regression runs.
- Ensure output head dimension and label preparation path are always coherent with regression for target outcomes.

#### 2) Best-model criterion alignment
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Keep best-checkpoint selection as minimization for regression metrics.
- Make monitored key explicit in result payload (e.g., `selection_metric: mse`).

### Success Criteria

#### Automated Verification:
- [x] CLI default reflects regression: `python dinov2/eval/linear3d_soop.py --help`
- [ ] Regression run for `gs_rankin_6isdeath` starts without classification branch usage.
- [ ] Regression run for `nihss` starts without classification branch usage.

#### Manual Verification:
- [ ] Team confirms no accidental binarization of `gs_rankin_6isdeath` in selected run configs.
- [ ] Run logs show regression pathway consistently for train/val/test.

**Implementation Note**: Pause after automated checks for human confirmation before Phase 6.

---

## Phase 6: Add complete regression metrics and persist `loss`

### Overview
Expand evaluation outputs to include the full regression metric set and persist train/val/test loss values in structured outputs.

### Changes Required

#### 1) Unified regression metric computation
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Add helper(s) to compute:
  - `mse = mean((y_hat - y)^2)`
  - `rmse = sqrt(mse)`
  - `mae = mean(|y_hat - y|)`
  - `mape = mean(|(y_hat - y) / max(eps, |y|)|) * 100`
  - `r2 = 1 - SS_res / max(eps, SS_tot)`
  - `loss` (regression objective, expected to align with MSE if MSELoss is retained)
- Handle degenerate cases safely (`eps`) to avoid division-by-zero or NaN propagation.

#### 2) Eval output schema extension
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Extend `val_metric`/`test_metric` payload to include a `metrics` dictionary containing all required keys.
- Keep backward compatibility for existing `name/value/samples` fields while adding rich metric fields.
- Persist aggregated train loss summary (e.g., last interval mean or epoch mean) to results JSON.

#### 3) Consistent logging lines
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Update logger output for eval checkpoints to print the full metric bundle.
- Keep compact formatting for frequent train iterations.

### Success Criteria

#### Automated Verification:
- [ ] JSON output contains keys `mae`, `mape`, `r2`, `rmse`, `mse`, `loss` under both val and test regression metrics.
- [ ] No NaN/Inf in metric values on smoke run.
- [ ] Existing checkpoint generation still works (`best_val_soop.pth`, periodic checkpoints).

#### Manual Verification:
- [ ] Metric values are numerically plausible relative to observed prediction ranges.
- [ ] `loss` and `mse` relationship is consistent with chosen objective.

**Implementation Note**: Pause after automated checks for human confirmation before Phase 7.

---

## Phase 7: Integrate W&B logging and API key export in runner

### Overview
Enable optional Weights & Biases experiment tracking for SOOP regression runs with explicit API-key export flow in bash runner.

### Changes Required

#### 1) W&B integration in eval entry
**File**: `baseline_encoder/3DINO/dinov2/eval/linear3d_soop.py`
**Changes**:
- Add CLI args for W&B controls (e.g., `--use-wandb`, `--wandb-project`, `--wandb-run-name`, `--wandb-tags`).
- Initialize W&B only on main process and only when enabled.
- Log:
  - train iteration stats (`loss`, `lr`),
  - periodic val metrics (full regression bundle),
  - final test metrics and run config summary.
- Ensure safe shutdown/finalize for W&B run.

#### 2) Dependency and docs update
**Files**:
- `baseline_encoder/3DINO/requirements*.txt` (or dependency file currently used in 3DINO env)
- `baseline_encoder/3DINO/README.md`
**Changes**:
- Add `wandb` dependency where 3DINO environment is declared.
- Document how to enable/disable W&B and required env vars.

#### 3) Export key + run metadata in bash runner
**File**: `baseline_encoder/3DINO/research/run_soop_outcome_experiments.sh`
**Changes**:
- Add explicit env handling pattern:
  - `WANDB_API_KEY` (required when `USE_WANDB=1`),
  - optional `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`, `WANDB_TAGS`.
- Export variables before command execution so each `conda run` inherits them.
- Pass W&B CLI flags into each run command in a consistent way.

### Success Criteria

#### Automated Verification:
- [x] `python dinov2/eval/linear3d_soop.py --help` shows W&B args.
- [x] Runner script passes shell syntax check: `bash -n research/run_soop_outcome_experiments.sh`.
- [ ] With `USE_WANDB=0`, runs execute exactly as local logging mode.
- [ ] With `USE_WANDB=1` and valid key, run initializes W&B successfully and logs metrics.

#### Manual Verification:
- [ ] W&B dashboard shows per-run config, val/test metrics, and final artifacts linkage.
- [ ] Multiple runs (image-only vs image+tabular, `nihss` vs `gs_rankin_6isdeath`) are distinguishable by run name/tags.

**Implementation Note**: Pause after this phase for final human review of dashboards and exported artifacts.

---

## Testing Strategy Addendum (Regression + W&B)

### Unit Tests
- Add focused tests for regression metric helper correctness (`mae`, `mape`, `r2`, `rmse`, `mse`) on deterministic tensors.
- Add edge-case tests for zero-target rows affecting `mape` denominator handling.

### Integration Tests
- Smoke run with `--task-type regression` and `--target-col nihss`.
- Smoke run with `--task-type regression` and `--target-col gs_rankin_6isdeath`.
- Smoke run with W&B disabled and enabled paths.

### Manual Testing Steps
1. Run regression-only `nihss` experiment and verify JSON + logs include full metric set.
2. Run regression-only `gs_rankin_6isdeath` experiment and verify no binary conversion occurs.
3. Run one W&B-enabled experiment and confirm metrics and config appear in dashboard.
