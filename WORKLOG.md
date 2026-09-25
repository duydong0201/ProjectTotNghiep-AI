# Worklog — ProjectTotNghiep-AI

> Ghi lại **chức năng / outcome đã hoàn thành** kèm bằng chứng kiểm chứng được. Mỗi ngày một mục, mới nhất ở trên.
> Không ghi cho từng commit nhỏ. Nhật ký tuần (khó khăn, bài học) nằm ở [JOURNAL.md](JOURNAL.md).

---

## [2026-09-25]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Phát hiện log v0 ghi theo sự kiện chứ không theo frame, làm hỏng nhãn `move` | Done | Đo trên 209.552 dòng log: 40.838 dòng (19,5%) mang nhãn `None` giả; ~45% frame vắng mặt. Ví dụ tái hiện: `Logs/2026-08-06_15-46-15.csv` frame 416 | - |
| duydong0201 | Bước tiền xử lý dựng lại frame cho log v0 | Done | `data.rebuild_frames_v0()`, `tests/test_data.py` (10 test); [ADR-004](docs/decisions/adr-004-rebuild-v0-frames.md); `check` PASS (ruff + 32 test) | - |

**Tổng kết ngày:** Nguyên nhân model `move` yếu không chỉ là thiếu bóng như kết luận ngày 21/09,
mà còn vì nhãn bị gán sai có hệ thống. Mọi số đo trong `reports/experiments.md` trước ngày này
không còn dùng để so sánh được, cần train lại. Việc tiếp theo: import lại log và chạy lại loạt
so sánh model trên nhãn đã sửa.

## [2026-09-21]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Phân tích repo game và đề xuất hướng Behavior Cloning | Done | [ADR-001](docs/decisions/adr-001-behavior-cloning.md), [PLAN.md](PLAN.md) | - |
| duydong0201 | Khởi tạo project AI: pipeline data → feature → train → export C++ | Done | `src/spike_ai/`; [docs/data_contract.md](docs/data_contract.md), [docs/integration.md](docs/integration.md) | - |
| duydong0201 | Vòng mỏng GĐ1 trên log v0 (31 trận, 60.637 frame) | Done (phía Python) | Model `move`: RF macro-F1 dev 0.543 vs baseline 0.272; `action` không học được do v0 thiếu bóng. Xem [experiments.md](reports/experiments.md). Chưa tích hợp vào game | - |
| duydong0201 | Exporter C++ riêng, có golden test | Done | [ADR-002](docs/decisions/adr-002-custom-cpp-exporter.md); `exports/spike_ai_move_decision_tree.h` (297 node). Chưa biên dịch thử bằng C++ | - |
| duydong0201 | Chia train/dev/holdout theo hash, có test chặn rò rỉ | Done | `src/spike_ai/split.py`, `tests/test_split.py`; `train --final` mới đo holdout | - |
| duydong0201 | Quy trình làm việc chắt lọc từ P-037: check script, CI, plan, ADR, worklog | Done | [docs/workflow.md](docs/workflow.md), `scripts/check.ps1`, `.github/workflows/ci.yml`; `check` PASS (ruff + 22 test) | - |

**Tổng kết ngày:** Có đầy đủ đường ống Python chạy trên dữ liệu thật và được kiểm thử. Kết quả trên log v0
chứng minh vấn đề đã dự đoán: không có thông tin bóng thì không học được `action`. Việc tiếp theo là chốt
schema v1 với bạn làm game, và thử header C++ trong game.
