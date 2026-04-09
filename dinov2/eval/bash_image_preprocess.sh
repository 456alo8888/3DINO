cd /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO
#wandb login 
# export WANDB_API_KEY=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
# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
#   --config-file dinov2/configs/train/vit3d_highres.yaml \
#   --pretrained-weights /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/dinov2/weight/3dino_vit_weights.pth \
#   --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/SOOP_result_image \
#   --dataset-name SOOP \
#   --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
#   --cache-dir /tmp/soop_cache \
#   --target-col gs_rankin_6isdeath \
#   --task-type regression \
#   --batch-size 32 \
#   --epochs 10 \
#   --epoch-length 125\
#   --eval-period-iterations 125\
#   --use-wandb \

export WANDB_API_KEY=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
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python dinov2/eval/linear3d_soop.py \
  --config-file dinov2/configs/train/vit3d_highres.yaml \
  --pretrained-weights /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/dinov2/weight/3dino_vit_weights.pth \
  --output-dir /mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO/SOOP_result_image \
  --dataset-name SOOP \
  --base-data-dir /mnt/disk2/hieupc2/Stroke_project/code/datasets/fold \
  --cache-dir /tmp/soop_cache \
  --target-col nihss \
  --task-type regression \
  --batch-size 32 \
  --epochs 10 \
  --epoch-length 125\
  --eval-period-iterations 125\
  --use-wandb \
  --wandb-project 3dino-soop-outcome-preprocessed \
  --wandb-run-name soop-preprocessed-nihss-image-only \
  --wandb-tags soop,raw,regression 