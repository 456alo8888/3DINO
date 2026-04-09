cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
export WANDB_API_KEY=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
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
#   --config-file dinov2/configs/train/vit3d_highres.yaml \
#   --pretrained-weights /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/dinov2/weight/3dino_vit_weights.pth \
#   --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/SOOP_result_image_tabular \
#   --dataset-name SOOP \
#   --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace \
#   --cache-dir /tmp/soop_raw_cache \
#   --target-col gs_rankin_6isdeath \
#   --task-type regression \
#   --use-tabular \
#   --batch-size 32 \
#   --epochs 10 \
#   --epoch-length 125 \
#   --eval-period-iterations 125 \
#   --num-workers 4 \
#   --optimizer auto \
#   --learning-rate 1e-3 \
#   --weight-decay 0.0 \
#   --grad-clip-norm 1.0 \
#   --diagnostics-period 10 \
#   --normalize-features \
#   --use-wandb \
#   --wandb-project 3dino-soop-outcome-raw \
#   --wandb-run-name soop-raw-gsrankin-image-tabular \
#   --wandb-tags soop,raw,regression \
#   --wandb-mode online

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/dinov2/weight/3dino_vit_weights.pth \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/SOOP_result_image_tabular \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold_raw_trace \
  --cache-dir /tmp/soop_raw_cache \
  --target-col nihss \
  --task-type regression \
  --use-tabular \
  --batch-size 32 \
  --epochs 10 \
  --epoch-length 125 \
  --eval-period-iterations 125 \
  --num-workers 4 \
  --optimizer auto \
  --learning-rate 1e-3 \
  --weight-decay 0.0 \
  --grad-clip-norm 1.0 \
  --diagnostics-period 10 \
  --normalize-features \
  --use-wandb \
  --wandb-project 3dino-soop-outcome-raw \
  --wandb-run-name soop-raw-nihss-image-tabular \
  --wandb-tags soop,raw,regression \
  --wandb-mode online