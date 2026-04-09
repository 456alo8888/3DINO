# Research Update: SOOP outcome pipeline (focus on regression for `nihss` and `gs_rankin_6isdeath`)

Date: 2026-03-19
Repository: `baseline_encoder/3DINO`

## 1) Tài liệu đã đọc để cập nhật bối cảnh
- `research/2026-03-19-3dino-soop-treatment-outcome.md`
- `research/plan.md`
- `research/implemented.md`

Các tài liệu trên khớp với code hiện tại: đã có dataset bridge SOOP, script eval riêng `linear3d_soop.py`, chế độ image-only và image+tabular.

## 2) Luồng dữ liệu hiện tại trong codebase

### 2.1 Nguồn nhãn và tạo split
- `preprocess_MRI/create_dataset/build_trace_splits.py`:
  - Merge từ participants/status/tabular.
  - Chuẩn hóa cột `nihss`, `gs_rankin_6isdeath` về numeric.
  - Giữ alias `gs_rankin+6isdeath = gs_rankin_6isdeath` trong manifest/split CSV.
- `datasets/fold/{train,valid,test}.csv` chứa nhãn đầu ra dùng cho huấn luyện/eval 3DINO.

### 2.2 Dataset gốc SOOP
- `datasets/SOOP_dataset.py`:
  - `DEFAULT_TARGET_COLS = ("nihss", "gs_rankin_6isdeath")`.
  - `_resolve_target_col(...)` hỗ trợ alias `gs_rankin+6isdeath`.
  - `SOOPTraceTabularDataset` trả về `image`, `tabular`, `target`, `label_mask` (và `subject_id` nếu bật).
  - `target` và `label_mask` luôn là vector theo danh sách target cols.

### 2.3 Adapter cho 3DINO downstream
- `dinov2/data/soop_dataset.py` (`SOOPOutcomeDataset`):
  - Chọn một target duy nhất bằng `target_col` + `target_index`.
  - Map sample về format downstream: `image`, `label`, `label_mask`, optional `tabular`, optional `subject_id`.
- `dinov2/data/loaders.py` (`make_classification_dataset_3d`, nhánh `dataset_name == 'SOOP'`):
  - Đọc `train.csv`, `valid.csv`, `test.csv` trong `base_directory`.
  - Tạo `SOOPOutcomeDataset` cho train/val/test.
  - Trả `class_num = 2` nếu `target_col` là `gs_rankin_6isdeath`/alias, ngược lại `class_num = 4`.

### 2.4 Transform cho SOOP
- `dinov2/data/transforms.py` đã có nhánh `dataset_name == 'SOOP'`.
- Pipeline dùng MONAI dict-transform trên key `image` và `label` (không remap label tại transform).

## 3) Hành vi task hiện tại trong `linear3d_soop.py`

### 3.1 CLI chính
- `--target-col`: chọn target (`nihss` hoặc `gs_rankin_6isdeath`/alias tương thích ở tầng dataset).
- `--task-type`: `binary | multiclass | regression`.
- `--drop-missing-labels`: bật lọc theo `label_mask`.
- `--use-tabular`: bật concat feature ảnh + vector tabular.

### 3.2 Chuyển đổi nhãn theo task type
Trong `_prepare_classification_labels(...)`:
- `regression`: giữ nhãn dạng liên tục `float`.
- `binary`: chuyển thành `(labels > 0).long()`.
- `multiclass`: `round` rồi `clamp` theo `num_classes`.

=> Nghĩa là nhị phân hóa không đến từ dữ liệu CSV, mà đến từ logic task ở eval script khi chọn `--task-type binary`.

### 3.3 Loss/metric theo task
- `_compute_loss(...)`:
  - regression: `MSELoss` (logits squeeze).
  - classification: `CrossEntropy`.
- `evaluate_model(...)`:
  - regression metric: `mse`.
  - classification metric: `accuracy`.
- Chọn best model:
  - classification: maximize metric.
  - regression: minimize metric.

### 3.4 Lọc missing label
- `_filter_valid_batch(...)` dùng `label_mask` nếu có.
- Khi `--drop-missing-labels` bật: chỉ giữ mẫu có mask > 0.
- Nếu tắt: giữ toàn bộ batch.

## 4) Trạng thái thực tế của nhãn `gs_rankin_6isdeath` trong dữ liệu hiện có
Đã kiểm tra trực tiếp CSV trong workspace:
- `datasets/fold/train.csv`, `valid.csv`, `test.csv`
- `preprocess_MRI/NoNaN_data.csv`
- `preprocess_MRI/processed_tabular/clinical_encoded.csv`

Kết quả thống nhất: `gs_rankin_6isdeath` hiện có giá trị `0..6` (không phải chỉ 0/1).

Tóm tắt số lượng:
- train: 461 mẫu, phân phối {0:72, 1:106, 2:66, 3:66, 4:92, 5:32, 6:27}
- valid: 99 mẫu, phân phối {0:15, 1:23, 2:14, 3:14, 4:20, 5:7, 6:6}
- test: 99 mẫu, phân phối {0:16, 1:23, 2:14, 3:14, 4:19, 5:7, 6:6}

## 5) Mapping hiện tại giữa dữ liệu và chế độ huấn luyện
- Cùng một cột `gs_rankin_6isdeath`:
  - Nếu `--task-type binary`: bị nhị phân hóa thành `0` vs `>0` trong code.
  - Nếu `--task-type regression`: giữ giá trị gốc liên tục (0..6).
- Cột `nihss`:
  - Có thể chạy classification hoặc regression tùy `--task-type`.
  - Với `regression`, nhãn giữ dạng liên tục.

## 6) Điểm nối kỹ thuật liên quan trực tiếp khi làm bài toán hồi quy
(Chỉ liệt kê vị trí hiện có trong code)
- Dataset/label mapping: `dinov2/data/soop_dataset.py`
- Dataset factory + split wiring: `dinov2/data/loaders.py`
- Task-type label conversion/loss/metric: `dinov2/eval/linear3d_soop.py`
- Nguồn cột nhãn + alias trong data pipeline: `datasets/SOOP_dataset.py`, `preprocess_MRI/create_dataset/build_trace_splits.py`

## 7) Kết luận trạng thái hiện tại
Codebase đã có đủ luồng cho cả classification và regression ở script SOOP downstream. Việc `gs_rankin_6isdeath` trở thành nhị phân hiện nay là do lựa chọn `--task-type binary` trong `linear3d_soop.py`, không phải do cột dữ liệu gốc trong CSV.