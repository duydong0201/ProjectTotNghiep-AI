# Kế hoạch thực hiện — AI học từ hành vi người chơi

> Lập ngày 21/09/2026. Mốc thời gian tính theo **tuần**, hãy điền ngày thật theo lịch
> bảo vệ đồ án của bạn. Đánh dấu `[x]` khi xong.

## Tình trạng (23/10/2026)

| GĐ | Tình trạng |
|---|---|
| 0-2 | Xong: pipeline, schema v1, `DatasetLogger`, export C++ + golden test |
| 3 | Một phần: 30 trận người thật + 298 trận tổng hợp (ADR-013); chưa đủ ≥ 8 người × 5 trận |
| 4-5 | Xong: Random Forest, label_shift (ADR-011), luật chặn, model **v10 vòng 2** trong game chính (model tự quyết nhảy) |
| 6 | Một phần: bot luật vs bot ML đo kỹ (v10 vòng 2: 80,3% điểm, thắng 100/100 trận). **Chưa:** người vs bot ML, holdout tương lai, độ giống người, khảo sát |
| 7 | DAgger + vòng tự học với chuyên gia (ADR-014, ADR-015). Chưa viết báo cáo, slide, video |

Các ô `[ ]` bên dưới là kế hoạch gốc, chưa đánh dấu lại từng ô.

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

## Định hướng phát triển: Hệ thống Bot AI đa cấp độ và đa phong cách chơi

Model Machine Learning hiện tại, huấn luyện bằng Behavior Cloning (kết hợp DAgger với chuyên gia quy tắc),
đã đạt hiệu quả cao: thắng bot Rule-based **100/100 trận** trong các thử nghiệm hiện tại (80,3% số điểm,
model `selfplay_expert_v10` vòng 2 – ADR-015). Đây là bot có năng lực cao nhất trong hệ thống, nhưng **chưa
được kiểm chứng đầy đủ với người chơi thật**.

Hướng phát triển tiếp theo là xây dựng **một hệ thống gồm nhiều bot AI có trình độ và phong cách chơi khác
nhau**, thay vì chỉ có một bot mạnh nhất:

- **Đa dạng trình độ:** các bot từ cơ bản đến nâng cao, khác nhau về khả năng tấn công, phòng thủ, di chuyển
  và phối hợp.
- **Đa dạng phong cách:** bot thiên về tấn công, bot phòng thủ tốt, bot di chuyển linh hoạt, hoặc bot có những
  điểm yếu chiến thuật nhất định.
- **Hệ thống xếp hạng Elo:** dùng kết quả thi đấu để ước lượng trình độ tương đối của các bot, hỗ trợ phân cấp
  và chọn đối thủ phù hợp.
- **Cá nhân hoá trải nghiệm:** khi có người chơi mới với trình độ khác nhau, game chọn bot phù hợp để tạo ra
  những trận đấu cân bằng và thú vị hơn.

**Mục tiêu cuối cùng:** chuyển từ **một model AI tối ưu sức mạnh** sang **một hệ sinh thái bot AI** có nhiều cấp
độ và phong cách, có thể đánh giá năng lực, phân loại trình độ và thích ứng với trải nghiệm của từng người chơi.

> **Lưu ý kỹ thuật:** Elo chủ yếu đo **sức mạnh tương đối**, không tự xác định phong cách chơi. Để tạo ra bot có
> phong cách riêng, cần thiết kế **dữ liệu huấn luyện, cách lấy mẫu hoặc mục tiêu huấn luyện** phù hợp cho từng
> kiểu bot.

### Cách làm dự kiến (tận dụng công cụ đã có)

| Việc | Cách làm | Công cụ sẵn có |
|---|---|---|
| Bot nhiều trình độ | Lấy các model ở các mốc phát triển (`models/01_…` → `15_…`), và chỉnh độ "kỹ" lúc chạy: ngưỡng ra tay (`ACT_THRESHOLD`), temperature rút thăm, tỉ lệ model tự quyết nhảy | `models/README.md`, `RunMode`, `AIInputSystem` |
| Bot nhiều phong cách | Đổi chuyên gia dạy (đập ngay / Set → Jump → Spike / thủ chắc), trọng số theo loại lượt, hoặc train theo dữ liệu một người chơi (`player_id`) | `ExpertDecide`, `expert_kill_weight`, `expert_quick_weight`, `controller_weight` |
| Đo phong cách | Mô tả mỗi bot bằng thống kê lối chơi (tỉ lệ đập ngay, chuỗi Set → Jump → Spike, Bump, đứng yên, cách mất điểm); có thể phân cụm K-means | `spike_ai.eval_ingame` |
| Xếp hạng Elo | Đấu vòng tròn giữa các bot bằng MatchRunner (không cửa sổ, ~1.000× thời gian thật, theo seed), cập nhật Elo sau mỗi trận | `MatchRunner.exe`, `spike_ai.selfplay` (build model riêng) |
| Ghép đối thủ cho người chơi | Ước lượng Elo người chơi sau vài trận, chọn bot có Elo gần nhất; cập nhật dần theo kết quả | Log `Logs/v1` + Elo của bot |

- [ ] Chọn 4–6 bot ở các trình độ / phong cách khác nhau, ghi cấu hình của từng bot
- [ ] Viết script đấu vòng tròn + tính Elo (seed cố định, đủ trận cho mỗi cặp)
- [ ] Bảng phong cách của từng bot từ `eval_ingame`
- [ ] Chọn bot theo Elo trong game (menu chọn độ khó hoặc tự ghép)
- [ ] Kiểm chứng với người chơi thật: trận có cân bằng hơn không, người chơi có thấy thú vị hơn không

---

## Phụ thuộc vào bạn làm game

| Việc | Cần trước | Ghi chú |
|---|---|---|
| Chốt schema v1 | GĐ0 | `docs/data_contract.md` |
| Hook cho `MLInputSystem` | GĐ1 | cùng interface với `AIInputSystem` |
| ~~`DatasetLogger` v1~~ | GĐ2 | ✅ 06/10/2026: `Source/System/DatasetLogger.cpp`, ghi mỗi frame một dòng vào `Logs/v1/<match_id>.csv`, gọi sau `GenerateIntent::update` và trước `ApplyIntentToComponent`. Đã kiểm chứng: 36 cột đúng schema, 0 frame trùng, 0 lỗ hổng, phía AI tính được 21 feature |
| Fixed timestep | GĐ2 | dùng `SystemConfig::FIXED_DT` thay `delta` thật để tái lập được |
| ~~Chế độ bot vs bot~~ | GĐ2 | ✅ 02/10/2026: `MatchRuleConfig::ENABLE_BOT_VS_BOT`, `AIInputSystem` đã tham số hoá theo phía sân |
| ~~Bỏ `std::rand()` khi chọn cú đập~~ | GĐ2 | ✅ 02/10/2026: `PickAttackIntent` chọn cú mạnh nhất còn rơi trong sân. Cần thiết để `spike_choice` học được (ADR-006) |

## Rủi ro

| Rủi ro | Cách giảm |
|---|---|
| Feature Python ≠ C++ | Spec JSON là nguồn sự thật + golden test ở GĐ1 |
| Ít dữ liệu người chơi | Bắt đầu thu từ tuần 4; dùng thêm dữ liệu bot; model đơn giản (tree) |
| Nhãn mất cân bằng (Spike rất hiếm) | class weight, lấy mẫu theo cửa sổ sự kiện, macro-F1 |
| Model offline tốt nhưng chơi dở (lỗi tích luỹ) | Đánh giá trong game sớm; action masking; DAgger |
| Game đổi format log | `schema_version` trong mỗi dòng; code từ chối version lạ |
