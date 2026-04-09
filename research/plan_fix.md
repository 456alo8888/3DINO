# Plan Fix: Ổn định huấn luyện SOOP regression trong `linear3d_soop.py`

## Overview

Mục tiêu của plan này là xử lý hiện tượng loss tăng đột biến sớm (ví dụ từ ~8.6 lên ~5.6e12 sau vài iteration) trong pipeline SOOP regression, đồng thời giữ kiến trúc hiện tại của 3DINO theo hướng **frozen backbone + train head**.

Plan này bám theo hiện trạng code đã có trong repo, chỉ đưa các thay đổi phù hợp:
- optimizer/regression head ổn định hơn,
- feature normalization,
- gradient clipping,
- guardrail cho pretrained weights,
- tăng quan sát train diagnostics.

## Current State Analysis

### Hiện trạng đã xác nhận từ code
- Backbone trong eval path đang chạy dưới `torch.inference_mode()` qua `ModelWithIntermediateLayers`, nên thực chất đã là head-only training.
- `linear3d_soop.py` đang dùng `SGD(lr=1e-3, momentum=0.9)` cho regression head.
- Regression loss là `MSELoss` trực tiếp trên output linear chưa có ràng buộc/chuẩn hóa feature.
- Chưa có gradient clipping trong vòng train linear eval.
- `setup.py` hiện cho phép fallback sang random initialization nếu không tìm thấy pretrained checkpoint.

### Giả thuyết nguyên nhân chính
1. Feature scale không được chuẩn hóa trước linear head -> gradient/loss nhạy với outlier.
2. SGD + momentum cho head regression trong bối cảnh feature scale không ổn định -> dễ overshoot sớm.
3. Không có clipping -> không chặn được gradient spike.
4. Nếu preload weight lỗi và rơi vào random-init path, đặc trưng backbone có thể không đủ ổn định cho regression probe.

## Desired End State

Sau khi hoàn tất plan:
1. Train regression head ổn định, không còn loss spike bất thường trong các iteration đầu.
2. Pipeline vẫn là frozen-backbone/head-only (không mở train backbone trong task này).
3. Có cơ chế guardrail để tránh vô tình chạy random-init khi expected là pretrained.
4. Có log chẩn đoán rõ ràng (feature norm, grad norm, optimizer config) để debug nhanh khi tái diễn.
5. Kết quả val/test metric (`mse`, `rmse`, `mae`, `mape`, `r2`, `loss`) cải thiện hoặc ít nhất không divergence.

### Key Discoveries
- `dinov2/eval/linear3d_soop.py`: optimizer hiện tại là SGD và chưa có grad clipping.
- `dinov2/eval/utils.py`: đã có utility `ModelWithNormalize` (có thể tận dụng ý tưởng normalize).
- `dinov2/eval/setup.py`: `FileNotFoundError` khi load weights đang được catch và tiếp tục chạy random-init.
- `dinov2/eval/segmentation3d.py`: có pattern freeze backbone rõ ràng bằng `requires_grad=False` (tham khảo để thêm assert/log).

## What We're NOT Doing

- Không thay đổi SSL pretraining (`dinov2/train/*`).
- Không thay đổi kiến trúc backbone ViT3D.
- Không thêm sweep framework lớn hay AutoML.
- Không mở train backbone full fine-tune trong scope fix này.

## Implementation Approach

Triển khai theo hướng incremental, có fallback rõ ràng:
1. Thêm guardrails + diagnostics trước (để biết chính xác điều gì đang xảy ra).
2. Áp dụng các thay đổi ổn định hóa ít xâm lấn nhất: feature normalization, gradient clipping.
3. Cho phép optimizer linh hoạt, mặc định ưu tiên phương án ổn định hơn cho regression head.
4. Chạy smoke + short-run theo cấu hình thống nhất, so sánh trước/sau bằng cùng seed.

---

## Phase 1: Guardrails & Observability

### Overview
Ngăn các lỗi cấu hình nguy hiểm và bổ sung đo đạc cần thiết để định vị nguyên nhân divergence trong runtime.

### Changes Required

#### 1) Strict pretrained guard
**File**: `dinov2/eval/setup.py`
**Changes**:
- Thêm cờ CLI (ví dụ: `--strict-pretrained`) trong setup parser.
- Khi bật strict mode: không được phép fallback random-init; phải raise lỗi rõ ràng nếu checkpoint không tồn tại hoặc không load được.

#### 2) Run metadata cho khả năng truy vết
**File**: `dinov2/eval/linear3d_soop.py`
**Changes**:
- Ghi thêm metadata vào `results_eval_linear_soop.json`:
  - optimizer type,
  - learning rate,
  - grad clip setting,
  - feature normalization bật/tắt,
  - strict-pretrained bật/tắt.

#### 3) Thêm diagnostics train-time
**File**: `dinov2/eval/linear3d_soop.py`
**Changes**:
- Log theo chu kỳ (ví dụ mỗi 10 iter):
  - `feature_absmax`, `feature_norm_mean`,
  - `pred_absmax`,
  - `grad_norm`.

### Success Criteria

#### Automated Verification:
- [x] `python dinov2/eval/linear3d_soop.py --help` hiển thị cờ strict/diagnostics mới.
- [x] Khi truyền đường dẫn weight sai + bật strict, process fail-fast với message rõ ràng.
- [x] JSON kết quả chứa đầy đủ trường metadata mới.

#### Manual Verification:
- [ ] Log đủ thông tin để phân biệt vấn đề do data scale, gradient spike, hay cấu hình weight.
- [ ] Team xác nhận fail-fast behavior không phá workflow hiện tại khi strict tắt.

**Implementation Note**: Dừng sau phase này để xác nhận guardrail/diagnostics trước khi thay đổi tối ưu hóa.

---

## Phase 2: Stabilize Head Optimization (Regression-first)

### Overview
Áp dụng các can thiệp ổn định hóa chính cho regression head nhưng giữ nguyên triết lý frozen backbone.

### Changes Required

#### 1) Optimizer selectable, default ổn định cho regression
**File**: `dinov2/eval/linear3d_soop.py`
**Changes**:
- Thêm cờ optimizer (ví dụ: `--optimizer {sgd,adamw}`), và tham số tương ứng (`--adamw-betas`, `--adamw-eps`).
- Đặt mặc định:
  - `adamw` cho `task-type=regression`,
  - vẫn cho phép override về `sgd` để ablation.

#### 2) Feature normalization trước head
**File**: `dinov2/eval/linear3d_soop.py`
**Changes**:
- Thêm cờ `--normalize-features`.
- Nếu bật: chuẩn hóa `img_feat` theo sample (L2 norm theo chiều feature) trước khi concat/tabular + linear head.
- Không thay đổi extraction backbone, chỉ xử lý tensor đầu vào head.

#### 3) Gradient clipping cho head
**File**: `dinov2/eval/linear3d_soop.py`
**Changes**:
- Thêm cờ `--grad-clip-norm` (0 nghĩa là tắt).
- Nếu >0: clip grad ngay sau `loss.backward()` và trước `optimizer.step()`.

#### 4) Explicit head-only assertion
**File**: `dinov2/eval/linear3d_soop.py`
**Changes**:
- Log số lượng trainable params của head và backbone.
- Assert backbone trainable params = 0 trong luồng eval để tránh vô tình unfreeze.

### Success Criteria

#### Automated Verification:
- [x] Train short-run (1 epoch, `epoch-length=20`) không xuất hiện loss explosion > 1e6 trong 20 iter đầu.
- [x] `grad_norm` nằm trong ngưỡng kỳ vọng khi bật clipping.
- [x] Chạy được cả 2 cấu hình:
  - `--optimizer adamw --normalize-features --grad-clip-norm 1.0`
  - `--optimizer sgd` (để đảm bảo backward compatibility).

#### Manual Verification:
- [ ] Curve train loss mượt hơn rõ rệt so với baseline diverging run.
- [ ] Team xác nhận frozen-backbone behavior không thay đổi so với trước fix.

**Implementation Note**: Dừng ở đây để chốt cấu hình mặc định tối ưu trước khi mở rộng runner/script.

---

## Phase 3: Experiment Script Alignment

### Overview
Đồng bộ script chạy thí nghiệm để sử dụng default ổn định mới và giữ khả năng so sánh ablation.

### Changes Required

#### 1) Cập nhật runner
**File**: `research/run_soop_outcome_experiments.sh`
**Changes**:
- Truyền thêm các cờ ổn định mặc định cho regression runs:
  - optimizer,
  - normalize-features,
  - grad-clip-norm,
  - strict-pretrained (khuyến nghị bật trong production run).

#### 2) Add baseline-vs-fix block
**File**: `research/run_soop_outcome_experiments.sh`
**Changes**:
- Bổ sung 1 block ablation ngắn để so sánh:
  - baseline cũ (SGD, no norm, no clip),
  - fixed config mới.

#### 3) Cập nhật docs vận hành
**Files**:
- `research/implemented.md`
- `research/problem_training.md`
**Changes**:
- Ghi rõ cấu hình ổn định đã áp dụng và lý do chọn.
- Ghi command tái lập divergence cũ + command fixed để team verify nhanh.

### Success Criteria

#### Automated Verification:
- [x] `bash -n research/run_soop_outcome_experiments.sh` pass.
- [ ] Runner tạo đủ output JSON/checkpoint cho cả baseline và fixed run.

#### Manual Verification:
- [ ] So sánh log cho thấy fixed run không còn nhảy loss bất thường như baseline.
- [ ] Team chấp nhận default mới cho các run regression tiếp theo.

**Implementation Note**: Dừng để xác nhận kết quả so sánh trước khi finalize.

---

## Phase 4: Validation & Acceptance

### Overview
Chốt nghiệm thu bằng bộ tiêu chí định lượng và checklist vận hành.

### Changes Required

#### 1) Validation matrix
**File**: `research/experiment_manifest.md`
**Changes**:
- Thêm mục validation matrix cho `gs_rankin_6isdeath` và `nihss` với 2 mode:
  - image-only,
  - image+tabular.
- Mỗi mode ghi rõ config stability đang dùng.

#### 2) Acceptance gates
**File**: `research/plan_fix.md` (this file)
**Changes**:
- Định nghĩa ngưỡng pass/fail cuối cùng cho merge.

### Success Criteria

#### Automated Verification:
- [ ] Tất cả run trong manifest hoàn tất không crash.
- [ ] Không có metric NaN/Inf trong `results_eval_linear_soop.json`.
- [ ] Regression metrics (`mse`, `rmse`, `mae`, `mape`, `r2`, `loss`) được ghi đầy đủ.

#### Manual Verification:
- [ ] Đường train loss của fixed config không còn explosion ở early iterations.
- [ ] Chất lượng val/test không tệ hơn baseline ổn định trước đó theo tiêu chí team chấp nhận.

---

## Testing Strategy

### Unit Tests
- Test hàm normalize feature (shape, finite values, không đổi batch size).
- Test nhánh grad clipping hoạt động khi `--grad-clip-norm > 0`.
- Test parser nhận đúng optimizer args mới.

### Integration Tests
- Smoke run regression với `epoch-length=2`, `eval-period-iterations=2` để check end-to-end.
- Short stability run với `epoch-length=20` để quan sát early-train dynamics.
- Run strict-pretrained fail-case với checkpoint path sai.

### Manual Testing Steps
1. Chạy baseline config cũ và lưu log/reference JSON.
2. Chạy fixed config mới cùng seed/dataset split.
3. So sánh train loss trajectory, grad norm trajectory, và val/test metrics.
4. Xác nhận runner + docs đủ để thành viên khác tái lập.

## Performance Considerations

- Feature normalization và gradient clipping làm tăng overhead rất nhỏ, phù hợp với linear-head training.
- AdamW trên head-only params có chi phí không đáng kể so với toàn pipeline.
- Diagnostics có thể giảm tần suất log để tránh IO bottleneck.

## Migration Notes

- Thay đổi là additive theo cờ CLI; có thể giữ backward compatibility bằng cách cho phép chọn lại SGD/no-norm/no-clip.
- Khuyến nghị team chuyển default regression runs sang profile ổn định mới sau khi phase 2 pass.

## References

- `research/problem_training.md`
- `research/research.md`
- `research/plan.md`
- `dinov2/eval/linear3d_soop.py`
- `dinov2/eval/setup.py`
- `dinov2/eval/utils.py`
- `dinov2/eval/linear3d.py`
- `dinov2/eval/segmentation3d.py`
