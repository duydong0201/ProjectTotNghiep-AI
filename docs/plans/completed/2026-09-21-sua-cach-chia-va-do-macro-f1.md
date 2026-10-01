# Sửa cách chia dữ liệu và cách đo macro-F1 — 2026-09-21

| Trường | Giá trị |
|---|---|
| Trạng thái | Completed |
| Branch | `fix/split-and-metrics` |
| Giai đoạn (PLAN.md) | GĐ1 (ảnh hưởng GĐ2, GĐ4) |
| Cập nhật lần cuối | 2026-09-21 17:10 |

## 1. Mục tiêu

Con số đánh giá phản ánh đúng chất lượng model:
- macro-F1 không bị kéo xuống bởi lớp vắng mặt trong tập đo;
- mỗi lần chia có báo cáo cân bằng và cảnh báo;
- dữ liệu người chơi được đánh giá bằng cross-validation, kèm holdout "tương lai" thu sau khi chốt model.

## 2. Phạm vi

- **Làm:**
  - Sửa `evaluate.py` (macro-F1 trên lớp có mặt, support, cảnh báo lớp hiếm).
  - Báo cáo cân bằng của mỗi lần chia.
  - Thêm chế độ đánh giá `single | kfold | lopo`.
  - Thêm `holdout_dirs` (holdout tương lai), kèm kiểm tra trùng trận.
  - Thêm cột `game_version` vào schema v1, kèm tuỳ chọn lọc theo phiên bản.
  - Cập nhật config, test, docs, ADR-004.
- **Không làm:** stratified split theo nhãn (phân tầng theo nhãn) trong phạm vi này.
- **Loại thay đổi:** cách chia dữ liệu; feature/schema (thêm cột log v1).

## 3. Các bước

- [x] `evaluate.py`: macro-F1 trên lớp có mặt trong tập đo; support từng lớp; danh sách lớp hiếm
- [x] `split.py`: `split_report` (số nhóm/frame, tỉ trọng nhóm lớn nhất, nhãn), cảnh báo; holdout_ratio = 0 hợp lệ
- [x] `train.py`: chế độ `single` / `kfold` / `lopo`; `holdout_dirs`; kiểm tra trùng trận; lọc `game_versions`
- [x] Config: v0 bỏ holdout; `human_v1` dùng kfold + holdout tương lai; thêm `human_v1_lopo`
- [x] Schema v1 thêm `game_version`; `docs/data_contract.md`
- [x] Test cho từng thay đổi; `scripts/check.ps1` PASS
- [x] ADR-004, `docs/workflow.md`, `WORKLOG.md`

## 4. Kiểm chứng

- Test: macro-F1 bỏ qua lớp vắng mặt; kfold/lopo không để nhóm lọt giữa các fold; holdout_dirs trùng trận thì báo lỗi.
- Chạy lại `baseline_v0`: in báo cáo cân bằng, không còn tập holdout.

## 5. Nhật ký tiến độ

| Thời gian | Nội dung | Quyết định / vướng mắc |
|---|---|---|
| 2026-09-21 16:00 | Tạo plan sau khi phân tích phân bố thật của v0 | Holdout v0 chỉ 5.2% frame, 78% từ một trận, thiếu SpikeLight |
| 2026-09-21 16:40 | Chạy lại v0 với `single` + `holdout_ratio: 0` | Cảnh báo mới bắt được: dev (hash < 0.2) trùng đúng holdout cũ, 1 trận chiếm 78% -> chuyển v0 sang `kfold` |
| 2026-09-21 16:50 | Lỗi `StringDtype` của pandas 3 trong CV với nhãn chuỗi | Sửa bằng cách copy `y_pool` rồi ghi đè từng fold; thêm `action` vào test kfold |
| 2026-09-21 17:00 | Cảnh báo lặp lại cho mỗi model, gây nhiễu | In cảnh báo chia một lần sau phần tóm tắt |

## 6. Tổng kết

> Chỉ điền khi Completed hoặc Cancelled.

- **Kết quả thực tế:** đủ phạm vi. v0 bằng CV 5 fold: RF `move` macro-F1 0.551 ± 0.040 (baseline 0.264 ± 0.007);
  `action` vẫn không học được (0.175 ± 0.008) vì v0 thiếu bóng.
- **Khác so với dự kiến:** v0 dùng `kfold` thay vì `single` như đề xuất ban đầu. Lý do: dev duy nhất bị một trận chiếm 78%.
  Thêm `human_v1_lopo.yaml`. Sửa tham chiếu cũ `tests/test_features.py` trong `data_contract.md`.
- **Lệnh kiểm chứng đã chạy và kết quả:** `bash scripts/check.sh` PASS (ruff + 45 test);
  `python -m spike_ai.train --config configs/baseline_v0.yaml`; export Decision Tree + golden test Python OK.
- **Việc tiếp theo:** báo bạn làm game về cột `game_version` và yêu cầu `match_id` duy nhất; lên lịch holdout tương lai ở GĐ6.
