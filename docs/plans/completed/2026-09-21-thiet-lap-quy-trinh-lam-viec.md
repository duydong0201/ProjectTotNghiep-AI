# Thiết lập quy trình làm việc và chia dữ liệu theo hash — 2026-09-21

| Trường | Giá trị |
|---|---|
| Trạng thái | Completed |
| Branch | N/A — commit khởi tạo repo trên `main`, sau đó mới tạo `develop` |
| Giai đoạn (PLAN.md) | GĐ0 |
| Cập nhật lần cuối | 2026-09-21 15:30 |

## 1. Mục tiêu

Repo có một quy trình làm việc rõ ràng, chắt lọc từ P-037, cho một người làm. Mỗi thay đổi có cổng kiểm tra
tự động (ở máy và CI). Cách chia dữ liệu tất định và không để lộ holdout.

## 2. Phạm vi

- **Làm:** `docs/workflow.md`, `AGENTS.md`, `scripts/check.{ps1,sh}` + ruff, CI GitHub Actions, template plan,
  ADR 001–003, `WORKLOG.md`, `JOURNAL.md`, `split.py` chia theo hash + test chặn rò rỉ.
- **Không làm:** tạo repo GitHub và push (cần tài khoản của người dùng); branch protection trên GitHub.
- **Loại thay đổi:** cách chia dữ liệu, dependency, quy trình.

## 3. Các bước

- [x] `split.py`: train/dev/holdout theo sha256 của khoá nhóm (trận hoặc người chơi); `train --final` mới đo holdout
- [x] `tests/test_split.py`: rời nhau, tất định, thêm dữ liệu không làm nhóm cũ đổi tập, salt bị khoá, `train` chỉ fit trên train
- [x] Ruff + `requirements-dev.txt` + `scripts/check.ps1` / `check.sh`
- [x] CI `.github/workflows/ci.yml` chạy trên push/PR vào `develop` và `main`
- [x] `docs/workflow.md`, `AGENTS.md`, template plan, ADR 001–003, `WORKLOG.md`, `JOURNAL.md`
- [x] Train lại trên log v0 với cách chia mới

## 4. Kiểm chứng

- `bash scripts/check.sh`: ruff check PASS, ruff format PASS, 22 test PASS.
- `python -m spike_ai.train --config configs/baseline_v0.yaml`: train 20 trận / dev 7 / holdout 4, holdout không được đo.

## 5. Nhật ký tiến độ

| Thời gian | Nội dung | Quyết định / vướng mắc |
|---|---|---|
| 2026-09-21 14:30 | Tạo plan | Không dùng owner-id / authorized scope của P-037 vì repo chỉ có một người làm |
| 2026-09-21 15:00 | Chia theo ngưỡng hash, không theo thứ hạng hash như P-037 | Ngưỡng hash giữ nguyên tập của nhóm cũ khi thêm dữ liệu, nhưng số lượng mỗi tập chỉ xấp xỉ tỉ lệ. Tập rỗng thì báo lỗi |
| 2026-09-21 15:20 | Ruff báo 7 lỗi (zip thiếu `strict`, tên biến `l`…) | Đã sửa, format lại 4 file |

## 6. Tổng kết

- **Kết quả thực tế:** hoàn thành đủ phạm vi. Kết quả dev của v0: RF `move` macro-F1 0.543 (baseline 0.272).
- **Khác so với dự kiến:** chia `requirements.txt` thành hai file (chạy pipeline / phát triển) để CI cài đúng thứ cần.
- **Lệnh kiểm chứng đã chạy và kết quả:** `bash scripts/check.sh` PASS (22 test).
- **Việc tiếp theo:** tạo repo GitHub, push `main` và `develop`; bật branch protection cho `develop`/`main` (tuỳ chọn).
