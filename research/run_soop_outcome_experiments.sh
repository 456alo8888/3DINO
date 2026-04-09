#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FOLD_DIR="${FOLD_DIR:-/mnt/disk2/hieupc2/Stroke_project/code/datasets/fold}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache/soop}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-path/to/eval/training_12499/teacher_checkpoint.pth}"
CONFIG_FILE="${CONFIG_FILE:-dinov2/configs/train/vit3d_highres.yaml}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-3dino-soop-outcome}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_TAGS="${WANDB_TAGS:-soop,regression}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-soop-regression}"
RUN_BASELINE_COMPARE="${RUN_BASELINE_COMPARE:-0}"

OPTIMIZER="${OPTIMIZER:-auto}"
NORMALIZE_FEATURES="${NORMALIZE_FEATURES:-1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
STRICT_PRETRAINED="${STRICT_PRETRAINED:-1}"

mkdir -p "$CACHE_DIR"

cd "$REPO_ROOT"

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  if [[ -z "$WANDB_API_KEY" ]]; then
    echo "[ERROR] USE_WANDB=1 nhưng WANDB_API_KEY chưa được set."
    exit 1
  fi
  export WANDB_API_KEY
  export WANDB_PROJECT
  export WANDB_ENTITY
  export WANDB_MODE
  export WANDB_TAGS
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-mode "$WANDB_MODE"
    --wandb-tags "$WANDB_TAGS"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

STABILITY_ARGS=(
  --optimizer "$OPTIMIZER"
  --grad-clip-norm "$GRAD_CLIP_NORM"
)
if [[ "$NORMALIZE_FEATURES" == "1" ]]; then
  STABILITY_ARGS+=(--normalize-features)
fi
if [[ "$STRICT_PRETRAINED" == "1" ]]; then
  STABILITY_ARGS+=(--strict-pretrained)
fi

if [[ "$RUN_BASELINE_COMPARE" == "1" ]]; then
  conda run -n hieupcvp env PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
    --config-file "$CONFIG_FILE" \
    --pretrained-weights "$PRETRAINED_WEIGHTS" \
    --output-dir "$REPO_ROOT/outputs/soop_gsrankin_image_only_baseline_sgd" \
    --dataset-name SOOP \
    --base-data-dir "$FOLD_DIR" \
    --cache-dir "$CACHE_DIR" \
    --target-col gs_rankin_6isdeath \
    --task-type regression \
    --optimizer sgd \
    --wandb-run-name "${WANDB_RUN_PREFIX}-baseline-gsrankin-image-only" \
    --batch-size 8 \
    --epochs 1 \
    --epoch-length 20 \
    --eval-period-iterations 20 \
    "${WANDB_ARGS[@]}"
fi

conda run -n hieupcvp env PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file "$CONFIG_FILE" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --output-dir "$REPO_ROOT/outputs/soop_gsrankin_image_only" \
  --dataset-name SOOP \
  --base-data-dir "$FOLD_DIR" \
  --cache-dir "$CACHE_DIR" \
  --target-col gs_rankin_6isdeath \
  --task-type regression \
  --wandb-run-name "${WANDB_RUN_PREFIX}-gsrankin-image-only" \
  --batch-size 8 \
  --epochs 1 \
  --epoch-length 20 \
  --eval-period-iterations 20 \
  "${STABILITY_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

conda run -n hieupcvp env PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file "$CONFIG_FILE" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --output-dir "$REPO_ROOT/outputs/soop_gsrankin_image_tabular" \
  --dataset-name SOOP \
  --base-data-dir "$FOLD_DIR" \
  --cache-dir "$CACHE_DIR" \
  --target-col gs_rankin_6isdeath \
  --task-type regression \
  --wandb-run-name "${WANDB_RUN_PREFIX}-gsrankin-image-tabular" \
  --use-tabular \
  --batch-size 8 \
  --epochs 1 \
  --epoch-length 20 \
  --eval-period-iterations 20 \
  "${STABILITY_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

conda run -n hieupcvp env PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file "$CONFIG_FILE" \
  --pretrained-weights "$PRETRAINED_WEIGHTS" \
  --output-dir "$REPO_ROOT/outputs/soop_nihss_regression" \
  --dataset-name SOOP \
  --base-data-dir "$FOLD_DIR" \
  --cache-dir "$CACHE_DIR" \
  --target-col nihss \
  --task-type regression \
  --wandb-run-name "${WANDB_RUN_PREFIX}-nihss-image-tabular" \
  --use-tabular \
  --batch-size 8 \
  --epochs 1 \
  --epoch-length 20 \
  --eval-period-iterations 20 \
  "${STABILITY_ARGS[@]}" \
  "${WANDB_ARGS[@]}"

echo "All SOOP outcome experiments finished."
