# Worklog — ProjectTotNghiep-AI

> Ghi lại **chức năng / outcome đã hoàn thành** kèm bằng chứng kiểm chứng được. Mỗi ngày một mục, mới nhất ở trên.
> Không ghi cho từng commit nhỏ. Nhật ký tuần (khó khăn, bài học) nằm ở [JOURNAL.md](JOURNAL.md).

---

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
