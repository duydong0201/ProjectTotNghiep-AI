# Worklog — ProjectTotNghiep-AI

> Ghi lại **chức năng / outcome đã hoàn thành** kèm bằng chứng kiểm chứng được. Mỗi ngày một mục, mới nhất ở trên.
> Không ghi cho từng commit nhỏ. Nhật ký tuần (khó khăn, bài học) nằm ở [JOURNAL.md](JOURNAL.md).

---

## [2026-10-01]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Gộp 3 nhánh đang treo vào `develop`, giải quyết trùng số ADR | Done | `develop` = merge của `fix/split-and-metrics`, `chore/pause-ci`, `fix/rebuild-v0-frames`; ADR trùng số 004 → đổi thành [ADR-005](docs/decisions/adr-005-rebuild-v0-frames.md); `check` PASS (ruff + 55 test) | - |
| duydong0201 | Import đủ log v0 và đo lại bằng cross-validation | Done | 101 trận / 241.542 frame. CV 5 fold, nhãn đã sửa: `move` RF **0.638 ± 0.029** (so cùng cách đo: 0.551 trước khi sửa nhãn, 31 trận). `action` vẫn 0.258 ± 0.137 = bằng majority. Xem [experiments.md](reports/experiments.md) run `20261001-233311` | - |

**Tổng kết ngày:** Đã có con số đáng tin đầu tiên cho `move`: Random Forest 0.638 ± 0.029 macro-F1, gấp
hơn hai lần baseline 0.297, và được đo bằng CV nên không phụ thuộc may rủi của một lần chia. KNN 0.639
nhúc nhích hơn nhưng chênh chưa tới một sai số chuẩn và không export sang C++ được, nên **chọn Random
Forest cho GĐ1**. `action` thì xác nhận dứt điểm là không học được ở v0 — độ lệch chuẩn (0.137) còn lớn
hơn khoảng cách tới majority, tức là hoàn toàn nhiễu. Việc tiếp theo: export RF ra header C++ và ghép
vào game để hoàn thành demo GĐ1.

## [2026-09-25]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Phát hiện log v0 ghi theo sự kiện chứ không theo frame, làm hỏng nhãn `move` | Done | Đo trên 209.552 dòng log: 40.838 dòng (19,5%) mang nhãn `None` giả; ~45% frame vắng mặt. Ví dụ tái hiện: `Logs/2026-08-06_15-46-15.csv` frame 416 | - |
| duydong0201 | Bước tiền xử lý dựng lại frame cho log v0 | Done | `data.rebuild_frames_v0()`, `tests/test_data.py` (10 test); [ADR-005](docs/decisions/adr-005-rebuild-v0-frames.md); `check` PASS (ruff + 32 test) | - |

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
| duydong0201 | Sửa cách chia dữ liệu và cách đo macro-F1 theo phân bố thật của v0 | Done | [ADR-004](docs/decisions/adr-004-data-split-strategy.md); CV 5 fold: RF `move` macro-F1 **0.551 ± 0.040** (dev 4 trận cũ báo 0.668 — sai lệch); holdout tương lai `human_holdout/`; `check` PASS (45 test) | - |

**Tổng kết ngày:** Có đầy đủ đường ống Python chạy trên dữ liệu thật và được kiểm thử. Kết quả trên log v0
chứng minh vấn đề đã dự đoán: không có thông tin bóng thì không học được `action`. Kiểm tra phân bố thật cho thấy
một tập dev/holdout duy nhất trên 31 trận không đáng tin, nên chuyển sang cross-validation (ADR-004). Việc tiếp theo là chốt
schema v1 với bạn làm game, và thử header C++ trong game.
