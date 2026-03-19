---
date: 2026-03-19T12:51:18+07:00
researcher: GitHub Copilot
git_commit: 85bd443
branch: main
repository: baseline_encoder/3DINO
topic: "Codebase mapping for applying 3DINO to treatment-outcome prediction with SOOP dataset"
tags: [research, codebase, 3dino, soop-dataset, treatment-outcome]
status: complete
last_updated: 2026-03-19
last_updated_by: GitHub Copilot
---

# Research: Codebase mapping for applying 3DINO to treatment-outcome prediction with SOOP dataset

**Date**: 2026-03-19T12:51:18+07:00  
**Researcher**: GitHub Copilot  
**Git Commit**: 85bd443  
**Branch**: main  
**Repository**: baseline_encoder/3DINO

## Research Question
How the codebase under `/mnt/disk2/hieupc2/Stroke_project/code/baseline_encoder/3DINO` is structured and how its implemented interfaces can be used for treatment-outcome prediction with `/mnt/disk2/hieupc2/Stroke_project/code/datasets/SOOP_dataset.py`.

## Summary
3DINO exposes a 3D ViT backbone pipeline for (1) self-supervised pretraining, (2) linear classification evaluation on frozen intermediate features, and (3) segmentation/visualization utilities. The model input path expects image tensors in 3D channel-first format and the downstream classification flow consumes image-label pairs from MONAI-style datalists. In the SOOP dataset module, each sample is a dictionary containing image, tabular vector, target vector, and label mask. The direct interface overlap is image tensor handling and batch collation, while outcome prediction targets map to the classification/evaluation side where labels are read from dataset records.

## Detailed Findings

### 1) 3DINO project entry points and structure
- Main project overview, dataset JSON formats, and run commands are documented in `README.md` ([baseline_encoder/3DINO/README.md:1](../../../../baseline_encoder/3DINO/README.md#L1), [baseline_encoder/3DINO/README.md:45](../../../../baseline_encoder/3DINO/README.md#L45), [baseline_encoder/3DINO/README.md:71](../../../../baseline_encoder/3DINO/README.md#L71), [baseline_encoder/3DINO/README.md:173](../../../../baseline_encoder/3DINO/README.md#L173)).
- Pretraining entry: `dinov2/train/train3d.py` main launcher ([baseline_encoder/3DINO/dinov2/train/train3d.py:340](../../../../baseline_encoder/3DINO/dinov2/train/train3d.py#L340)).
- Classification downstream entry: `dinov2/eval/linear3d.py` ([baseline_encoder/3DINO/dinov2/eval/linear3d.py:1](../../../../baseline_encoder/3DINO/dinov2/eval/linear3d.py#L1)).
- Segmentation downstream entry: `dinov2/eval/segmentation3d.py` ([baseline_encoder/3DINO/dinov2/eval/segmentation3d.py:348](../../../../baseline_encoder/3DINO/dinov2/eval/segmentation3d.py#L348)).
- Feature visualization entry: `dinov2/eval/vis_pca.py` ([baseline_encoder/3DINO/dinov2/eval/vis_pca.py:177](../../../../baseline_encoder/3DINO/dinov2/eval/vis_pca.py#L177)).

### 2) Model construction and feature APIs
- Evaluation model is built from config, teacher weights are loaded, then set to eval+CUDA in `build_model_for_eval` ([baseline_encoder/3DINO/dinov2/eval/setup.py:63](../../../../baseline_encoder/3DINO/dinov2/eval/setup.py#L63)).
- Setup helper returns model and autocast dtype in `setup_and_build_model_3d` ([baseline_encoder/3DINO/dinov2/eval/setup.py:75](../../../../baseline_encoder/3DINO/dinov2/eval/setup.py#L75)).
- `ModelWithIntermediateLayers` wraps backbone intermediate extraction via `get_intermediate_layers(..., return_class_token=True)` ([baseline_encoder/3DINO/dinov2/eval/utils.py:31](../../../../baseline_encoder/3DINO/dinov2/eval/utils.py#L31)).
- Dataset-wide feature extraction utilities exist in `extract_features_dict` and `extract_features_with_dataloader` ([baseline_encoder/3DINO/dinov2/eval/utils.py:149](../../../../baseline_encoder/3DINO/dinov2/eval/utils.py#L149), [baseline_encoder/3DINO/dinov2/eval/utils.py:164](../../../../baseline_encoder/3DINO/dinov2/eval/utils.py#L164)).

### 3) Tensor shape and input expectations in backbone
- Core 3D backbone class: `DinoVisionTransformer3d` ([baseline_encoder/3DINO/dinov2/models/vision_transformer.py:300](../../../../baseline_encoder/3DINO/dinov2/models/vision_transformer.py#L300)).
- Token preparation unpacks shape as `B, C, W, H, D` in `prepare_tokens_with_masks` ([baseline_encoder/3DINO/dinov2/models/vision_transformer.py:357](../../../../baseline_encoder/3DINO/dinov2/models/vision_transformer.py#L357)).
- Intermediate feature export for downstream heads is implemented in `get_intermediate_layers` ([baseline_encoder/3DINO/dinov2/models/vision_transformer.py:368](../../../../baseline_encoder/3DINO/dinov2/models/vision_transformer.py#L368)).

### 4) Downstream classification data path in 3DINO
- Classification datasets are created through `make_classification_dataset_3d` using datalist JSON splits and MONAI `PersistentDataset` ([baseline_encoder/3DINO/dinov2/data/loaders.py:154](../../../../baseline_encoder/3DINO/dinov2/data/loaders.py#L154)).
- Generic pretraining dataset path is `make_dataset_3d` reading JSON list records ([baseline_encoder/3DINO/dinov2/data/loaders.py:49](../../../../baseline_encoder/3DINO/dinov2/data/loaders.py#L49)).
- Classification transform definitions are in `make_classification_transform_3d` with MONAI dictionary transforms over `image` and `label` ([baseline_encoder/3DINO/dinov2/data/transforms.py:31](../../../../baseline_encoder/3DINO/dinov2/data/transforms.py#L31)).
- Linear probing head consumes concatenated class tokens (and optional pooled patch tokens) in `create_linear_input` ([baseline_encoder/3DINO/dinov2/eval/linear3d.py:150](../../../../baseline_encoder/3DINO/dinov2/eval/linear3d.py#L150)).

### 5) SOOP dataset module interfaces relevant to adaptation
- Main dataset class is `SOOPTraceTabularDataset` ([datasets/SOOP_dataset.py:108](../../../../datasets/SOOP_dataset.py#L108)).
- Per-sample output dictionary from `__getitem__` contains `image`, `tabular`, `target`, `label_mask`, and optionally `subject_id` ([datasets/SOOP_dataset.py:212](../../../../datasets/SOOP_dataset.py#L212)).
- Split builders and dataloader builders are provided in `build_soop_trace_datasets` and `build_soop_trace_dataloaders` ([datasets/SOOP_dataset.py:273](../../../../datasets/SOOP_dataset.py#L273)).
- LMDB-backed dataset variant is implemented in `SOOPTraceTabularLMDBDataset` ([datasets/SOOP_dataset.py:369](../../../../datasets/SOOP_dataset.py#L369)).

### 6) Existing interface connection points (3DINO ↔ SOOP)
- `SOOPTraceTabularDataset` already returns image tensors compatible with 3D model consumption (channel-first 3D volume path), while 3DINO downstream utilities operate on image-label inputs.
- 3DINO evaluation utilities and linear probing expect model features derived from image tensors; SOOP’s `target`/`label_mask` fields provide outcome labels/masks that can be selected/converted into the label path used by classification pipelines.
- SOOP has explicit split CSV tooling and optional LMDB access in the same module, while 3DINO uses JSON datalist conventions for MONAI datasets.

## Code References
- `baseline_encoder/3DINO/README.md:45-58` - pretraining datalist schema (`image`, `shape`, `spacing`).
- `baseline_encoder/3DINO/README.md:152-170` - finetuning datalist schema (`training`/`validation`/`test`, each with `image` and `label`).
- `baseline_encoder/3DINO/dinov2/eval/setup.py:63-73` - eval model construction and pretrained weight loading.
- `baseline_encoder/3DINO/dinov2/eval/utils.py:31-45` - intermediate feature extraction wrapper.
- `baseline_encoder/3DINO/dinov2/eval/utils.py:149-197` - batch-wise feature bank extraction.
- `baseline_encoder/3DINO/dinov2/models/vision_transformer.py:357-384` - 3D token prep and intermediate layer output structure.
- `baseline_encoder/3DINO/dinov2/data/loaders.py:154-225` - classification dataset loader creation.
- `baseline_encoder/3DINO/dinov2/data/transforms.py:31-147` - classification transform pipeline.
- `datasets/SOOP_dataset.py:108-248` - tabular+image sample contract and missing-label mask handling.
- `datasets/SOOP_dataset.py:369-419` - LMDB sample contract.

## Architecture Documentation
- 3DINO architecture layers are organized under `dinov2/models`, `dinov2/layers`, and task-specific eval modules under `dinov2/eval`.
- Training and evaluation are config-driven, with CLI entry points and helper setup utilities.
- Data interface in 3DINO is dictionary-style MONAI records; downstream classification operates on image tensors and categorical/label targets through evaluation wrappers.
- SOOP dataset architecture centralizes CSV-based multimodal samples and can expose either filesystem-backed or LMDB-backed sample access.

## Historical Context (from thoughts/)
- No pre-existing `thoughts/` documents were present in this workspace before this research run.

## Related Research
- No additional research documents found prior to this report.

## Open Questions
- Which outcome target encoding from SOOP (`nihss`, `gs_rankin_6isdeath`, or derived labels) will be used as the supervised label field for the 3DINO linear-eval style pipeline.
- Whether the downstream path will consume image-only supervision or combine image features with `tabular` vectors from SOOP in a separate head.
