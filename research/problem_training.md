# Problem Analysis: Loss tăng đột biến khi train SOOP regression

**Date**: 2026-03-19  
**Repo**: `baseline_encoder/3DINO`  
**Scope**: Document current codepath and where divergence manifests in training.

## 1) Triệu chứng quan sát được

Từ log train:
- `iter=0 loss=8.656578`
- `iter=10 loss=5588756987904.000000`

Loss tăng đột biến rất lớn trong vài iteration đầu.

## 2) Command path đang dùng

Trong `dinov2/eval/bash.sh`, command chạy:
- `target-col=gs_rankin_6isdeath`
- `task-type=regression`
- optimizer mặc định trong script là SGD (`learning-rate=1e-3`, momentum `0.9`)

## 3) Điểm trong code nơi loss tăng được tạo ra

### 3.1 Loss regression là raw MSE trên output tuyến tính không bị chặn
- File: `dinov2/eval/linear3d_soop.py`
- `_compute_loss(...)` dùng:
  - `mse_loss(logits.squeeze(-1), labels.float())`
- Head regression là 1 lớp linear, output không có activation giới hạn (`SOOPLinearHead.forward` trả trực tiếp `self.linear(feat)`).

=> Khi output linear tăng độ lớn, MSE tăng theo bình phương sai số, nên có thể bùng rất nhanh.

### 3.2 Vòng tối ưu làm loss divergence xuất hiện trực tiếp ở mỗi step
- File: `dinov2/eval/linear3d_soop.py`
- Trong train loop:
  1. `logits = head(img_feat, tabular)`
  2. `loss = _compute_loss(...)`
  3. `loss.backward()`
  4. `optimizer.step()`

Loss được log trực tiếp bằng `loss.item()` (không smoothing, không clipping).

## 4) Bằng chứng về scale dữ liệu đầu vào/nhãn

Đã kiểm tra các split CSV (`datasets/fold/train.csv`, `valid.csv`, `test.csv`):
- `gs_rankin_6isdeath`: min `0`, max `6`
- `nihss`: min `0`, max `35`
- Tabular candidate columns: max absolute value ~`89` (cột `age`)

=> Không thấy nhãn/tabular có biên độ bất thường kiểu hàng triệu tỷ để tự sinh MSE ~1e12 ngay từ bản thân target.

## 5) Các yếu tố liên quan trong current implementation

### 5.1 Feature extraction path
- `ModelWithIntermediateLayers` chạy backbone dưới autocast (`fp16/bf16` tùy config), sau đó `create_linear_input(...)` cast output về `float()`.
- Không có bước chuẩn hóa L2 feature trong `linear3d_soop.py`.

### 5.2 Optimizer/scheduler path
- Optimizer hiện tại: `torch.optim.SGD(..., lr=1e-3, momentum=0.9)`.
- Scheduler: cosine annealing theo `epochs * epoch_length`.

### 5.3 Pretrained loading behavior
- Trong `dinov2/eval/setup.py`, nếu `pretrained_weights` không tìm thấy sẽ bắt `FileNotFoundError` và tiếp tục chạy với random initialization.
- Đây là nhánh hiện có trong code và ảnh hưởng trực tiếp tới độ ổn định nếu đường dẫn weight không hợp lệ.

## 6) Kết luận vị trí vấn đề trong codebase

Loss tăng đột biến biểu hiện tại nhánh train regression trong `dinov2/eval/linear3d_soop.py`, cụ thể tại chuỗi:
- linear head output không ràng buộc
- MSE loss trực tiếp trên output đó
- cập nhật SGD theo từng batch

Dữ liệu nhãn/tabular trong split hiện tại không cho thấy scale bất thường; do đó điểm divergence nằm ở động học tối ưu trên output/feature path của training loop hiện tại hơn là do target values thô trong CSV.

## 7) Cập nhật sau khi áp dụng fix (2026-03-19)

Các thay đổi đã được triển khai trong code:
- `strict-pretrained` guard để fail-fast khi checkpoint không hợp lệ.
- `optimizer` selectable (`auto|sgd|adamw`) với regression mặc định đi theo nhánh ổn định hơn (`adamw` khi dùng `auto`).
- `normalize-features` (L2 theo sample trước linear head).
- `grad-clip-norm` cho head sau `backward`.
- Diagnostics theo chu kỳ: `grad_norm`, `feature_absmax`, `feature_norm_mean`, `pred_absmax`.

Kết quả smoke ngắn với profile fix (`adamw + normalize + clip`):
- `iter=0 loss=2.283813`, `grad_norm=3.27`
- `iter=10 loss=3.013942`, `grad_norm=3.87`
- Không xuất hiện bùng nổ loss kiểu `1e12` như log ban đầu.

Kết quả kiểm tra tương thích với `optimizer=sgd` vẫn chạy được, nhưng cho thấy gradient/loss tăng mạnh nhanh hơn:
- `iter=0 loss=7.358904`, `grad_norm=327.93`
- `iter=1 loss=109.336159`, `grad_norm=1209.29`

=> Cấu hình fix profile hiện là lựa chọn khuyến nghị cho regression runs trong SOOP.
