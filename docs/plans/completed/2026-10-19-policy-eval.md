# Đo offline đúng cách bot ra quyết định trong game — 2026-10-19

| Trường | Giá trị |
|---|---|
| Trạng thái | Completed |
| Branch | `feat/policy-eval` |
| Giai đoạn (PLAN.md) | GĐ4 / GĐ6 |
| Cập nhật lần cuối | 2026-10-19 |

## 1. Mục tiêu

Có một con số offline nói được về **con bot thật trong game**: áp đúng cách ra quyết định của
`AIInputSystem.cpp` (ngưỡng P(ra tay), rút thăm có temperature, các luật chặn, luật nhảy sau
Set) lên **toàn bộ frame** của các trận không dùng để train, và so với cách đo hiện tại
(argmax trên cửa sổ ±20 frame).

## 2. Phạm vi

- **Làm:** `src/spike_ai/policy.py` (cách ra quyết định), `src/spike_ai/eval_policy.py` (CLI đo
  theo CV theo trận), test, ghi kết quả vào `reports/experiments.md`.
- **Không làm:** phần luật gốc của bot (di chuyển, `PickAttackIntent`) và cú đập trên không sau khi
  bot tự nhảy - cả hai phụ thuộc vào vị trí CỦA BOT, mà dữ liệu offline đi theo vị trí người chơi.
  Lọc "cú đập phải rơi trong sân" (`SpikeLandsInCourt`) cũng chưa chép, vì cần mô phỏng quỹ đạo.
- **Loại thay đổi:** code (`src/`, `tests/`, `docs/`).

## 3. Các bước

- [x] `policy.py`: phân phối hành động của bot cho từng frame, tính chính xác (không rút thăm)
- [x] `eval_policy.py`: CV theo trận, train trên cửa sổ như config, đo trên mọi frame
- [x] Test + `scripts/check.ps1` PASS
- [x] Chạy trên `played_v1_aligned`, ghi `reports/experiments.md`
- [x] `WORKLOG.md`

## 4. Kiểm chứng

- Test tự động: ngưỡng, temperature, chặn Jump/Serve, phủ quyết ngoài vùng detect, luật nhảy,
  ma trận nhầm lẫn kỳ vọng.
- Chỉ số: bảng so sánh 3 cách đo trên cùng các fold.

## 5. Nhật ký tiến độ

| Thời gian | Nội dung | Quyết định / vướng mắc |
|---|---|---|
| 2026-10-19 | Tạo plan | Dùng phân phối kỳ vọng thay cho rút thăm để kết quả không phụ thuộc seed |

## 6. Tổng kết

- **Kết quả thực tế:** có `eval_policy`. 330 trận: macro-F1 argmax/cửa sổ 0,411 → bot trong game 0,173; 30 trận thật: 0,108 → 0,141.
- **Khác so với dự kiến:** phát hiện 298/338 file `Logs/v1` là dữ liệu sinh tự động có vật lý không khớp game (ADR-013). Công thức vùng detect đã kiểm chứng đúng trên dữ liệu thật.
- **Lệnh kiểm chứng:** `scripts/check.ps1` PASS (119 test); `python -m spike_ai.eval_policy --config configs/played_v1_aligned.yaml`.
- **Việc tiếp theo:** người dùng quyết định tách dữ liệu sinh tự động khỏi dữ liệu chơi thật.
