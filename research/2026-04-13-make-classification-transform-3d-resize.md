---
date: 2026-04-13T00:54:04+07:00
researcher: GitHub Copilot
git_commit: de927c8
branch: main
repository: 3DINO
topic: "How make_classification_transform_3d resizes image input"
tags: [research, codebase, transforms, resize, soop]
status: complete
last_updated: 2026-04-13
last_updated_by: GitHub Copilot
---

# Research: How make_classification_transform_3d resizes image input

**Date**: 2026-04-13T00:54:04+07:00  
**Researcher**: GitHub Copilot  
**Git Commit**: de927c8  
**Branch**: main  
**Repository**: 3DINO

## Research Question
Read the codebase, especially make_classification_transform_3d in transforms.py, to document how image input is resized.

## Summary
The resize behavior in make_classification_transform_3d is dataset-dependent and always applied after foreground cropping.

- For ICBM and SOOP, resize is controlled by image_size.
: image_size == 0 uses identity (no resizing); otherwise the image is resized to a cubic target (image_size, image_size, image_size).
- For COVID-CT-MD, the pipeline first resizes to a fixed size (144, 144, 112), then crops to (image_size, image_size, image_size).
: training uses random crop, validation uses center crop.

## Detailed Findings

### Transform Entry Point
- The function is defined in [dinov2/data/transforms.py](dinov2/data/transforms.py#L55).
- It receives dataset_name, image_size, and min_int, then builds MONAI Compose pipelines for train and validation.

### Shared Resize Primitive Used in ICBM and SOOP
- Global resize selector near function start:
  - [dinov2/data/transforms.py](dinov2/data/transforms.py#L68): identity when image_size == 0.
  - [dinov2/data/transforms.py](dinov2/data/transforms.py#L70): Resized to (image_size, image_size, image_size) when image_size != 0.
- In ICBM:
  - train uses resize_transform at [dinov2/data/transforms.py](dinov2/data/transforms.py#L92).
  - val uses resize_transform at [dinov2/data/transforms.py](dinov2/data/transforms.py#L116).
- In SOOP:
  - SOOP branch defines the same conditional resize behavior at [dinov2/data/transforms.py](dinov2/data/transforms.py#L171) and [dinov2/data/transforms.py](dinov2/data/transforms.py#L173).
  - train applies resize_transform at [dinov2/data/transforms.py](dinov2/data/transforms.py#L183).
  - val applies resize_transform at [dinov2/data/transforms.py](dinov2/data/transforms.py#L205).

### COVID-CT-MD Resize + Crop Strategy
- Fixed pre-resize to (144, 144, 112):
  - train [dinov2/data/transforms.py](dinov2/data/transforms.py#L138)
  - val [dinov2/data/transforms.py](dinov2/data/transforms.py#L163)
- Then target-size crop to cube based on image_size:
  - train uses random crop [dinov2/data/transforms.py](dinov2/data/transforms.py#L139)
  - val uses center crop [dinov2/data/transforms.py](dinov2/data/transforms.py#L164)

### SOOP Input Shape Normalization Before Resize
- SOOP pipelines run channel normalization before resize using _ensure_soop_channel_first:
  - helper definition [dinov2/data/transforms.py](dinov2/data/transforms.py#L32)
  - used in train [dinov2/data/transforms.py](dinov2/data/transforms.py#L178)
  - used in val [dinov2/data/transforms.py](dinov2/data/transforms.py#L200)
- Behavior documented by code:
  - 3D input (H, W, D) becomes (1, H, W, D).
  - 4D channel-last singleton (H, W, D, 1) is moved to channel-first.
  - 4D channel-first singleton (1, H, W, D) is preserved.

### Where the Transform Is Used
- Runtime call sites of make_classification_transform_3d:
  - [dinov2/eval/linear3d.py](dinov2/eval/linear3d.py#L471)
  - [dinov2/eval/linear3d_soop.py](dinov2/eval/linear3d_soop.py#L366)

### Downstream SOOP Flow (context for resized tensor)
- SOOP datasets are constructed with these transforms in [dinov2/data/loaders.py](dinov2/data/loaders.py#L194), [dinov2/data/loaders.py](dinov2/data/loaders.py#L201), and [dinov2/data/loaders.py](dinov2/data/loaders.py#L209).
- The transform is applied per sample in [dinov2/data/soop_dataset.py](dinov2/data/soop_dataset.py#L82).
- Dataset output enforces shape (1, H, W, D) before batching in [dinov2/data/soop_dataset.py](dinov2/data/soop_dataset.py#L64) and [dinov2/data/soop_dataset.py](dinov2/data/soop_dataset.py#L66).
- Batches are consumed as image tensors in [dinov2/eval/linear3d_soop.py](dinov2/eval/linear3d_soop.py#L476).

## Code References
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L55) - make_classification_transform_3d definition
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L68) - no-resize path (Identityd)
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L70) - cubic resize path (Resized)
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L138) - COVID fixed pre-resize in train
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L139) - COVID random crop to image_size
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L163) - COVID fixed pre-resize in val
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L164) - COVID center crop to image_size
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L171) - SOOP no-resize path
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L173) - SOOP cubic resize path
- [dinov2/data/transforms.py](dinov2/data/transforms.py#L32) - SOOP channel-first normalization helper
- [dinov2/eval/linear3d.py](dinov2/eval/linear3d.py#L471) - call site in linear eval flow
- [dinov2/eval/linear3d_soop.py](dinov2/eval/linear3d_soop.py#L366) - call site in SOOP eval flow

## Architecture Documentation
Current transform architecture uses dataset-specific branching within one function:
- ICBM and SOOP share a resize selector driven by image_size.
- COVID-CT-MD applies a fixed canonical resize before final target-size crop.
- SOOP additionally normalizes channel placement prior to intensity/crop/resize operations.

## Historical Context (from repository research notes)
- Existing notes that describe the broader SOOP pipeline:
  - [research/research.md](research/research.md)
  - [research/2026-03-19-3dino-soop-treatment-outcome.md](research/2026-03-19-3dino-soop-treatment-outcome.md)

## Related Research
- [research/research.md](research/research.md)
- [research/2026-03-19-3dino-soop-treatment-outcome.md](research/2026-03-19-3dino-soop-treatment-outcome.md)

## Open Questions
- None for the requested scope.
