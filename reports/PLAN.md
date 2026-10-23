# KẾ HOẠCH DỰ TÍNH PHÁT TRIỂN AI BEHAVIOR CLONING (PLAN.MD)

> **Tài liệu tham chiếu:** [BAO-CAO-TIEN-DO-1.md](file:///f:/DoAnTotNghiep/BAO-CAO-TIEN-DO-1.md) — _Mục 11: Trạng thái hiện tại và bước tiếp theo_.
> **Dự án:** Game Simulation Runtime & AI Behavior Cloning (The Spike Cross Remaster).
> **Mục tiêu cốt lõi:** Xây dựng hệ thống AI học từ hành vi người chơi (Behavior Cloning) có khả năng tự điều khiển nhân vật thi đấu trọn vẹn một trận đấu bóng chuyền trong game.

---

## 1. TỔNG QUAN LỘ TRÌNH (ROADMAP PROGRESSION)

Theo cấu trúc từ Mục 11 của báo cáo tiến độ, lộ trình phát triển được phân định qua các trạng thái:

```text
┌────────────────────────────────────────────────────────┐
│ [ĐÃ XONG]  Nền tảng Pipeline & Baseline Spike Model    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ [ĐANG LÀM] GIAI ĐOẠN 1: Logger v1 & Thu thập Dữ liệu   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ [KẾ TIẾP]  GIAI ĐOẠN 2: Kiểm chứng Bot & Action Model  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ [KẾ TIẾP]  GIAI ĐOẠN 3: Tích hợp Toàn diện             │
│            (Action + Move + Spike)                     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ [MỤC TIÊU] GIAI ĐOẠN 4: AI Tự Chơi Trọn Vẹn 1 Trận     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼ (Nếu còn thời gian)
┌────────────────────────────────────────────────────────┐
│ [MỞ RỘNG]  GIAI ĐOẠN 5: DAgger & Cá nhân hóa Model     │
└────────────────────────────────────────────────────────┘
```

---

## 2. CHI TIẾT TỪNG GIAI ĐOẠN & CÔNG VIỆC CỤ THỂ

### Giai đoạn 0: Nền tảng đã hoàn thành [ĐÃ XONG]

_Đã hoàn tất kiểm nghiệm kỹ thuật, làm cơ sở cho các bước tiếp theo:_

- [x] **Behavior Cloning Pipeline:** Chu trình khép kín Game → Logger → Preprocessing → Train → Eval → C++ Export → Runtime.
- [x] **Model Random Forest Spike:** Phân loại 3 mức lực đập bóng (`SpikeLight`, `SpikeMedium`, `SpikeStrong`).
- [x] **Đánh giá & Validation:** Group-wise 5-Fold Cross Validation theo `match_id`, đo lường bằng Macro-F1.
- [x] **Export C++ & Golden Test:** Xuất decision tree sang mã C++ header độc lập, kiểm chứng sai số inference giữa Python và C++ đạt chuẩn `< 1e-5`.

---

### Giai đoạn 1: Chuẩn hóa Dữ liệu & Logger v1 [ĐANG LÀM]

_Thời gian dự tính: Tuần 1 – Tuần 2_
_Mục tiêu: Khắc phục triệt để lỗi thiếu trạng thái quả bóng và sai lệch nhãn từ phiên bản Logger v0._

#### Các đầu việc cụ thể:

1. **Hoàn thiện Dataset Logger v1 trong Game C++:**
   - **Ghi 1 dòng/frame (Per-frame logging):** Không chỉ ghi khi có sự kiện bấm phím; ghi liên tục mọi frame (kể cả frame không có hành động để model học nhãn `None`).
   - **Ghi State trước Intent:** Ghi nhận trạng thái tại thời điểm quyết định thay vì sau khi intent đã biến đổi simulation (loại bỏ data leakage / nhãn trễ).
   - **Bổ sung Ball State (yếu tố quyết định):**
     - Vị trí và thông số bóng: `ball_x`, `ball_y`, `ball_a`, `ball_b`, `ball_c`.
     - Dự đoán điểm rơi: `ball_landing_x`, `ball_state_frame`.
   - **Bổ sung Rally & Action State:**
     - Trạng thái pha bóng: `rally_last_touch`, `rally_touch_count`.
     - Trạng thái nhân vật: `p_action_state`, `p_action_remain_ms`, `p_input_move`, `p_input_intent`.
     - Metadata: `match_id`, `player_id`, `controller`, `game_version`.

2. **Thu thập dữ liệu người chơi bổ sung (Data Collection):**
   - Thi đấu bổ sung **15 – 20 trận** có chủ đích (người chơi chủ động thực hiện đa dạng kỹ thuật).
   - | Cân bằng số lượng mẫu cho các class thiểu số: | Loại Intent |   Hiện tại (v0)    | Mục tiêu tối thiểu (v1) |
     | :-------------------------------------------- | :---------: | :----------------: | ----------------------- |
     | `SpikeStrong`                                 |      5      |        150         |
     | `SpikeLight`                                  |     17      |        150         |
     | `SpikeMedium`                                 |     19      |        150         |
     | `Set` (Chuyền 2)                              |     55      |        150         |
     | `Jump` (Nhảy)                                 |      0      |        150         |
     | `Bump` (Đệm bóng)                             |     774     | Duy trì chất lượng |
   - Tự động hóa kiểm tra số lượng mẫu qua script `count_spikes.py` sau mỗi phiên chơi.

_Tiêu chuẩn nghiệm thu (DoD Giai đoạn 1):_

- File log chuẩn v1: Không bị mất frame liên tục, không có giá trị NaN, các trường về quả bóng có dữ liệu hợp lệ 100%.
- Số mẫu `SpikeStrong` và `Jump` đạt ngưỡng tối thiểu ≥ 150 mẫu mỗi loại.

---

### Giai đoạn 2: Kiểm chứng Pipeline & Huấn luyện Action Model [KẾ TIẾP]

_Thời gian dự tính: Tuần 3 – Tuần 4_
_Mục tiêu: Đảm bảo pipeline học tốt trên dữ liệu có luật rõ ràng trước khi huấn luyện dữ liệu người chơi, sau đó xây dựng mô hình ra quyết định hành động._

#### Các đầu việc cụ thể:

1. **Kiểm chứng Pipeline bằng Opponent Bot (Sanity Verification):**
   - Thu thập dữ liệu từ các trận đấu của Bot có tập luật cố định (Rule-based Bot).
   - Huấn luyện mô hình học lại hành vi của Rule-based Bot.
   - **Tiêu chí bắt buộc:** Đạt **Macro-F1 > 0.90**.
   - _Quy tắc:_ Nếu không đạt `> 0.90`, lập tức dừng lại để debug Logger, Feature extraction và Labeling; không chuyển sang dữ liệu người chơi khi pipeline chưa qua bài test này.

2. **Xây dựng kiến trúc mô hình phân tầng (Hierarchical Action Model):**
   - **Tầng A (Action Decision Model):**
     - Giải quyết câu hỏi: _"Ở frame này có nên hành động không, và là hành động gì?"_ (Phân loại giữa `None` và ~9 hành động chính).
     - Áp dụng kỹ thuật lấy mẫu cục bộ quanh thời điểm chạm bóng (Contact-window sampling) để giảm áp lực mất cân bằng của lớp `None`.
   - **Tầng B (Spike Intensity Model):**
     - Nếu Tầng A quyết định `Spike` → Tầng B xác định mức lực: `Light`, `Medium`, hoặc `Strong`.
     - Tái sử dụng và nâng cấp model Random Forest đã được chứng minh ở Giai đoạn 0.
   - **Mô hình Di chuyển (Move Model):**
     - Dự đoán giá trị di chuyển ngang `moveX` (-1: Trái, 0: Đứng yên, 1: Phải) dựa trên vị trí tương quan với bóng và lưới.

3. **Tối ưu hóa mô hình:**
   - Điều chỉnh siêu tham số Random Forest (`class_weight='balanced'`, `min_samples_leaf`).
   - Đánh giá trên tập holdout độc lập (không chạm vào holdout trong quá trình tinh chỉnh).

_Tiêu chuẩn nghiệm thu (DoD Giai đoạn 2):_

- Pipeline bot đạt Macro-F1 > 0.90.
- Model Tầng A vượt trội so với `majority baseline`.
- Model Tầng B (Spike) duy trì F1 trên cả 3 lớp (bao gồm `SpikeStrong` > 0.6).

---

### Giai đoạn 3: Tích hợp Hệ thống (Action + Move + Spike) [KẾ TIẾP]

_Thời gian dự tính: Tuần 5 – Tuần 6_
_Mục tiêu: Ghép nối các mô hình thành một khối điều khiển AI hoàn chỉnh chạy trực tiếp trong C++ Simulation Engine._

#### Các đầu việc cụ thể:

1. **Export đồng bộ các Sub-models sang C++:**
   - Cập nhật exporter tự động chuyển đổi cả 3 mô hình (Action Model, Move Model, Spike Model) thành các header C++ độc lập, không phụ thuộc thư viện bên ngoài.
   - Chạy bộ kiểm thử Golden Test để đối chiếu đầu ra giữa Python và C++ cho toàn bộ các model.

2. **Xây dựng bộ điều phối suy luận trong Game (`MLInputSystem` / `AIInputSystem`):**
   - Hiện thực hóa sơ đồ ra quyết định:
     ```text
     State Game (Mỗi Tick)
            │
            ▼
     [Move Model] ───► Sinh Move Intent (Trái/Phải/Đứng yên)
            │
            ▼
     [Action Model - Tầng A]
            │
       Có hành động?
      ├── KHÔNG ───► Duy trì di chuyển/chờ
      └── CÓ ──────► Xác định loại Action (Set / Bump / Jump / Spike / ...)
                        │
                        ▼ (Nếu là Spike)
                 [Spike Model - Tầng B]
                        │
                        ▼
                 Lực: Light / Medium / Strong
                        │
                        ▼
                 Tổng hợp thành Final Intent đẩy vào Simulation Core
     ```

3. **Đồng bộ hóa nhịp điều khiển (Tick Rate):**
   - Đảm bảo AI suy luận ăn khớp với `FIXED_DT` (60 FPS) của simulation core mà không làm sụt giảm hiệu năng rendering.

_Tiêu chuẩn nghiệm thu (DoD Giai đoạn 3):_

- Mã nguồn C++ tích hợp biên dịch sạch (Clean build), không có memory leak trong vòng lặp game loop.
- Golden Test C++ đạt độ chính xác tương đương Python trên toàn bộ các mô hình.

---

### Giai đoạn 4: Thử nghiệm Thực tế & Đạt Mục tiêu [MỤC TIÊU CỐT LÕI]

_Thời gian dự tính: Tuần 7_
_Mục tiêu: AI tự vận hành và chơi trọn vẹn một trận đấu thực tế trong game._

#### Các đầu việc cụ thể:

1. **Kiểm tra năng lực thi đấu cơ bản (Gameplay Capability):**
   - Khả năng đỡ bóng cơ bản (Bump) khi bóng sang sân.
   - Khả năng đón bóng chuyền (Set) và bật nhảy đập bóng (Spike) đúng thời điểm.
   - Không bị đứng yên (freeze) hoặc chạy vào góc sân không mục đích.
2. **Đánh giá định tính & định lượng:**
   - Tỷ lệ hoàn thành rally (số chạm bóng thành công trung bình mỗi điểm).
   - Tỷ lệ lỗi tự phát sinh (đánh trượt bóng ngớ ngẩn do suy luận sai trạng thái).
   - AI thi đấu liên tục từ đầu trận đến khi kết thúc (đủ điểm thắng/thua) mà không bị crash.

_Tiêu chuẩn nghiệm thu (DoD Giai đoạn 4 - Cột mốc hoàn thành đồ án):_

- **AI hoàn thành trọn vẹn 1 trận đấu không cần can thiệp thủ công.**
- Thể hiện được các hành vi đặc trưng học từ người chơi (biết phối hợp đỡ - chuyền - đập).

---

### Giai đoạn 5: Mở rộng & Tối ưu nâng cao [NẾU CÒN THỜI GIAN]

_Thời gian dự tính: Tuần 8+ (Dành cho phần điểm cộng đồ án)_
_Mục tiêu: Nâng cao độ thông minh và khả năng thích ứng của bot._

#### Các đầu việc cụ thể:

1. **Nghiên cứu & Ứng dụng DAgger (Dataset Aggregation):**
   - Khắc phục hiện tượng trôi trạng thái (Covariate Shift) khi AI vào các tình huống lạ ngoài tập dữ liệu huấn luyện.
   - Cho AI tự chơi → Ghi lại các frame AI xử lý lỗi → Chuyên gia/Người chơi điều khiển sửa lỗi tại các frame đó → Bổ sung vào tập huấn luyện và train lại.
2. **Cá nhân hóa phong cách chơi (Player-Specific Models):**
   - Tách log của từng người chơi cụ thể để train model riêng (ví dụ: Bot lối chơi tấn công nhanh vs. Bot lối chơi phòng thủ an toàn).
3. **Thử nghiệm LightGBM:**
   - Triển khai thử nghiệm nếu Random Forest bị giới hạn độ chuẩn xác, xây dựng bộ C++ Tree Inference riêng cho GBDT.

---

## 3. BẢNG TIẾN ĐỘ DỰ TÍNH (TIMELINE MATRIX)

| Giai đoạn       | Tuần thực hiện | Trọng tâm công việc                                      | Đầu ra (Deliverable)                        |   Trạng thái   |
| :-------------- | :------------: | :------------------------------------------------------- | :------------------------------------------ | :------------: |
| **Giai đoạn 0** |     Đã qua     | Pipeline BC, Random Forest Spike, C++ Export             | Golden test C++ pass, model Spike v0        | **[ĐÃ XONG]**  |
| **Giai đoạn 1** |       -        | Hoàn thiện Logger v1, thu thập 15-20 trận người chơi     | Schema v1, Log dataset v1 (≥150 mẫu/class)  | **[ĐANG LÀM]** |
| **Giai đoạn 2** |     **-**      | Sanity test Bot (F1>0.9), train Action Model tầng A + B  | Baseline bot validated, Hierarchical Models | **[KẾ TIẾP]**  |
| **Giai đoạn 3** |     **-**      | Export sang C++, tích hợp Action + Move + Spike vào game | `MLInputSystem` C++ hoàn chỉnh              | **[KẾ TIẾP]**  |
| **Giai đoạn 4** |       -        | Thử nghiệm AI tự chơi toàn trận, cân chỉnh gameplay      | **AI tự chơi trọn vẹn 1 trận đấu**          | **[MỤC TIÊU]** |
| **Giai đoạn 5** |  **Tuần 4+**   | Áp dụng DAgger, cá nhân hóa model, viết báo cáo tổng kết | Báo cáo hoàn chỉnh, DAgger pipeline         | **[MỞ RỘNG]**  |

---

## 4. QUẢN TRỊ RỦI RO & PHƯƠNG ÁN DỰ PHÒNG (RISK MITIGATION)

| Rủi ro tiềm ẩn                                             |    Mức độ    | Biện pháp xử lý                                                                                                     |
| :--------------------------------------------------------- | :----------: | :------------------------------------------------------------------------------------------------------------------ |
| **Dữ liệu vẫn mất cân bằng lớn sau khi có v1**             |     Cao      | Áp dụng lấy mẫu quanh thời điểm chạm bóng (Contact-window sampling); điều chỉnh`class_weight='balanced'`.           |
| **Sanity test trên Opponent Bot không đạt F1 > 0.9**       | Nghiêm trọng | Dừng ngay việc train human data; rà soát sai lệch tọa độ bóng, nhãn trễ frame giữa C++ simulation và Python logger. |
| **Model dự đoán trượt nhịp khi chạy real-time trong game** |  Trung bình  | Kiểm tra biến`FIXED_DT`, đảm bảo inference tính bằng microsecond; dùng state trước intent để triệt tiêu độ trễ.     |
| **AI rơi vào vòng lặp hành vi (Covariate Shift)**          |  Trung bình  | Áp dụng DAgger (Giai đoạn 5) hoặc bổ sung rule fallback nếu AI rơi vào trạng thái ngoài biên sân.                   |

---

## 5. NGUYÊN TẮC ĐÁNH GIÁ ĐỊNH KỲ

1. **Luôn so sánh với Majority Baseline:** Bất kỳ mô hình nào được chọn phải có Macro-F1 vượt trội so với việc chỉ đoán lớp đa số.
2. **Sử dụng Macro-F1 làm thước đo chính:** Không dùng Accuracy đơn thuần vì dữ liệu game thể thao luôn bị chi phối bởi các trạng thái chờ (`None`).
3. **Giữ tập Holdout độc lập:** Tập test cuối cùng tuyệt đối không được dùng để điều chỉnh siêu tham số hay lựa chọn feature.
