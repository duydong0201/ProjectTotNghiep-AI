# Kế hoạch thực hiện — AI học từ hành vi người chơi

> Lập ngày 21/09/2026. Mốc thời gian tính theo **tuần**, hãy điền ngày thật theo lịch
> bảo vệ đồ án của bạn. Đánh dấu `[x]` khi xong.

## Tổng quan

```text
GĐ0 Chuẩn bị ─► GĐ1 Vòng mỏng (v0) ─► GĐ2 Data v1 + bot ─► GĐ3 Thu data người
                                                              │
      GĐ7 Báo cáo ◄─ GĐ6 Đánh giá trong game ◄─ GĐ5 Tích hợp ◄─ GĐ4 Train & chọn model
                              │                                      ▲
                              └──────── kết quả chưa tốt ────────────┘
```

Nguyên tắc: **làm một vòng đầu-cuối thật sớm (GĐ1)**, sau đó mới cải thiện từng khâu.
Lỗi nguy hiểm nhất là feature tính trong Python khác với trong C++, và chỉ phát hiện được
khi đã tích hợp vào game.

---

## GĐ0 — Chuẩn bị (Tuần 1)

- [ ] Tạo venv, cài `requirements-dev.txt`, `scripts/check.ps1` PASS
- [ ] Đọc code game: `AIInputSystem`, `CharacterIntent`, `FinalIntent`, `ActionState`,
      `MainScene::update`, `RallyStateUtils::LogTrajectoryEvent`
- [ ] Đọc `docs/data_contract.md`, **họp với bạn làm game để chốt schema v1**
- [ ] Tạo repo GitHub `ProjectTotNghiep-AI`, push commit đầu tiên

**Xong khi:** hai bên đã thống nhất `schema/feature_spec.v1.json` (có thể còn chỉnh nhỏ).

## GĐ1 — Vòng mỏng end-to-end trên log v0 (Tuần 2)

Mục tiêu là chứng minh đường ống thông suốt, **không cần model tốt**.

- [ ] `scripts/import_logs.py` → copy log hiện có vào `data/raw/v0`
- [ ] `python -m spike_ai.train --config configs/baseline_v0.yaml` chạy được
- [ ] Export Decision Tree sang header C++ (`spike_ai.export_cpp`)
- [ ] Cùng bạn làm game tạo `MLInputSystem` gọi header đã sinh, `RunGoldenTest()` trả về true
- [ ] Bot do model điều khiển di chuyển được trong game (dù còn "ngu")

**Xong khi:** thấy bot ML chạy trong game và golden test C++ khớp với Python.

## GĐ2 — Dữ liệu v1 + tái tạo bot rule-based (Tuần 3–4)

*Phụ thuộc bên game: logger v1, fixed timestep (`FIXED_DT`) thay cho `delta` thật.*

- [ ] Bên game implement `DatasetLogger` theo schema v1
- [ ] Viết test kiểm tra file v1 (đủ cột, đúng kiểu, frame liên tục, không NaN)
- [ ] Cho bot đánh với bot / với người vài chục trận → `data/raw/v1/bot`
- [ ] Train model bắt chước `AIInputSystem` (config `configs/baseline_v1.yaml`)
- [ ] **Sanity check:** bot hành xử theo luật cố định nên macro-F1 phải cao (> 0.9 với `move`).
      Nếu thấp → lỗi nằm ở feature/log, sửa trước khi làm tiếp.

**Xong khi:** model v1 bắt chước bot rule-based tốt, cả offline lẫn trong game.

## GĐ3 — Thu thập dữ liệu người chơi (Tuần 4–6, làm song song GĐ2/GĐ4)

- [ ] Viết hướng dẫn ngắn cho người chơi, xin đồng ý dùng dữ liệu
- [ ] Mục tiêu: ≥ 8 người, mỗi người ≥ 5 trận; ghi `player_id` ẩn danh (P01, P02…)
- [ ] Lưu vào `data/raw/v1/human`, backup lên Google Drive
- [ ] Notebook `01_eda.ipynb`: phân bố nhãn, độ dài trận, heatmap vị trí, khác biệt giữa người chơi

**Xong khi:** có đủ dữ liệu và biết rõ các vấn đề của nó (mất cân bằng, nhiễu, trận lỗi).

## GĐ4 — Clean, feature, train và chọn model (Tuần 6–8)

- [ ] Clean: bỏ trận lỗi/bỏ dở, bỏ frame giữa lúc reset
- [ ] Căn nhãn: trạng thái frame t ↔ input frame t; thử thêm độ trễ phản xạ (t−k)
- [ ] Mất cân bằng: `class_weight`, giảm frame `None`, chỉ lấy cửa sổ quanh lúc chạm bóng
- [ ] Feature engineering (chỉ những gì viết lại được trong C++, ghi vào spec)
- [ ] Train và so sánh: majority → Decision Tree → Random Forest → (LightGBM) → (MLP)
- [ ] Đánh giá: macro-F1, confusion matrix, **CV 5 fold theo trận** (`configs/human_v1.yaml`),
      và **leave-one-player-out** (`configs/human_v1_lopo.yaml`), xem ADR-004
- [ ] Chọn model theo: macro-F1, tốc độ suy luận, dễ export, tính deterministic

**Xong khi:** có bảng so sánh trong `reports/experiments.md` và lý do chọn model.

## GĐ5 — Export & tích hợp (Tuần 8–9)

- [ ] Export model được chọn (`spike_ai.export_cpp` cho cây, hoặc ONNX nếu dùng MLP)
- [ ] `MLInputSystem` trong game: tính feature → gọi model → chặn hành động không hợp lệ
      (action masking) → xuất `CharacterIntent`
- [ ] Golden test: 20 vector đầu vào cho ra cùng kết quả ở Python và C++
- [ ] Đo thời gian suy luận mỗi frame (mục tiêu < 0.1 ms)

## GĐ6 — Đánh giá trong game (Tuần 9–11)

- [ ] Thu **holdout tương lai**: các phiên chơi mới sau khi đã chốt model, có ≥ 2 người chưa từng chơi
      → `data/raw/v1/human_holdout/` → `train --final` (ADR-004)

- [ ] Tỷ lệ thắng: ML-bot vs rule-bot, ML-bot vs người (≥ 20 trận mỗi cặp)
- [ ] Độ giống người: so sánh phân bố hành động, vị trí đứng, thời điểm đánh bóng
- [ ] Khảo sát người chơi (Likert 1–5): "bot này chơi giống người không?", có thể làm blind test
- [ ] Phân tích lỗi: bot ML hay sai ở tình huống nào? → quay lại GĐ4 nếu cần

## GĐ7 — Mở rộng & báo cáo (Tuần 11–12+)

Chọn 1–2 mục nếu còn thời gian:
- [ ] Phân cụm phong cách chơi (K-means trên thống kê mỗi trận)
- [ ] Bot bắt chước riêng từng người chơi (train theo `player_id`)
- [ ] DAgger để giảm lỗi tích luỹ của behavior cloning
- [ ] Adaptive difficulty

Báo cáo:
- [ ] Chương dữ liệu (schema, quy trình thu thập, EDA)
- [ ] Chương phương pháp (behavior cloning, feature, model)
- [ ] Chương thực nghiệm (bảng `experiments.md`, confusion matrix, đánh giá trong game)
- [ ] Slide + video demo

---

## Phụ thuộc vào bạn làm game

| Việc | Cần trước | Ghi chú |
|---|---|---|
| Chốt schema v1 | GĐ0 | `docs/data_contract.md` |
| Hook cho `MLInputSystem` | GĐ1 | cùng interface với `AIInputSystem` |
| `DatasetLogger` v1 | GĐ2 | ghi mỗi frame, trước `ApplyIntentToComponent` |
| Fixed timestep | GĐ2 | dùng `SystemConfig::FIXED_DT` thay `delta` thật để tái lập được |
| Chế độ bot vs bot | GĐ2 | để tự sinh dữ liệu và đánh giá hàng loạt |

## Rủi ro

| Rủi ro | Cách giảm |
|---|---|
| Feature Python ≠ C++ | Spec JSON là nguồn sự thật + golden test ở GĐ1 |
| Ít dữ liệu người chơi | Bắt đầu thu từ tuần 4; dùng thêm dữ liệu bot; model đơn giản (tree) |
| Nhãn mất cân bằng (Spike rất hiếm) | class weight, lấy mẫu theo cửa sổ sự kiện, macro-F1 |
| Model offline tốt nhưng chơi dở (lỗi tích luỹ) | Đánh giá trong game sớm; action masking; DAgger |
| Game đổi format log | `schema_version` trong mỗi dòng; code từ chối version lạ |
