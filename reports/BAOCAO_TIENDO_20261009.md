# Báo cáo tiến độ — Mô hình AI học từ hành vi người chơi

**Ngày:** 09/10/2026
**Đề tài:** Xây dựng AI điều khiển nhân vật bóng chuyền bằng cách học từ dữ liệu người chơi
(Behavior Cloning), tích hợp vào game Axmol C++.

---

## 1. Đã làm được gì

| # | Hạng mục | Trạng thái |
|---|----------|-----------|
| 1 | Pipeline dữ liệu: đọc log CSV → canonical hoá theo phía sân → sinh feature | Xong |
| 2 | Logger trong game ghi đủ trạng thái bóng (schema v1, 36 cột) | Xong |
| 3 | Huấn luyện + đánh giá bằng cross-validation chia theo trận (GroupKFold) | Xong |
| 4 | Xuất model Random Forest sang header C++ (không phụ thuộc thư viện ngoài) | Xong |
| 5 | Port 21 feature sang C++, đối chiếu khớp 100% với Python | Xong |
| 6 | Tích hợp: model điều khiển bot, có lớp action masking bằng luật | Xong |
| 7 | Đo thực nghiệm trong game: bot luật vs bot ML, cùng một người chơi | Xong |

### Hai điểm kỹ thuật đáng kể nhất

**a) Phát hiện và sửa lỗi nhãn ở log v0.** Logger cũ chỉ ghi khi có sự kiện, nên khi
người chơi *giữ* nút di chuyển thì nhãn bị nhân bản sai, và khoảng 45% frame không được
ghi ra. Đo được 19,5% nhãn sai. Sau khi dựng lại chuỗi frame, điểm `move` tăng
0,529 → 0,624.

**b) Kiểm chứng hai lớp giữa Python và C++.**

- *Feature-level:* 21 feature tính ở C++ khớp Python **0/21 sai lệch** trên 3.512 frame,
  cho cả hai phía sân.
- *Model-level:* golden test 20 vector chạy mỗi lần khởi động game — C++ phải cho cùng
  kết quả với Python, nếu lệch thì báo lỗi ngay.

Nhờ bước này, khi bot chơi sai ta biết chắc nguyên nhân là **model / dữ liệu**, không
phải lỗi tích hợp.

---

## 2. Thông số đánh giá (offline, cross-validation chia theo trận)

Chỉ số dùng: **macro-F1** (bình quân F1 của từng lớp). Không dùng accuracy vì dữ liệu
mất cân bằng cực đoan — đoán toàn `None` cho accuracy 0,971 nhưng macro-F1 chỉ 0,123.

### 2.1 Bước nhảy lớn nhất: thêm cột trạng thái bóng (v0 → v1)

| Target | Baseline (đoán lớp đông nhất) | v0 (không có bóng) | v1 (có bóng) |
|---|---|---|---|
| `action` (làm gì) | 0,123 | 0,258 ≈ baseline | **0,911** |
| `move` (đi hướng nào) | 0,273 | 0,624 | **0,765** |

Kết luận: không có trạng thái bóng thì bài toán `action` **không học được** (0,258 bằng
đúng mức đoán bừa). Đây là lý do phải viết lại logger sang schema v1. Sau khi có bóng,
`action` đạt 0,911 = **7,4 lần** baseline.

### 2.2 Kiểm tra tổng quát hoá — LOPO (train 3 người, test người thứ 4 chưa từng thấy)

| Target | K-Fold | LOPO | Mất |
|---|---|---|---|
| `action` | 0,911 | 0,908 | 0,003 |
| `move` | 0,765 | 0,761 | 0,004 |

Gần như không mất điểm ⇒ model học **quy luật chung**, không học thuộc từng người.
(Lưu ý trung thực: 4 người chơi có thể còn khá giống nhau nên con số này có thể dễ dãi.)

### 2.3 So sánh thuật toán (bộ v1, target `action`)

| Model | macro-F1 | Ghi chú |
|---|---|---|
| Majority (baseline) | 0,123 | mốc so sánh |
| Logistic Regression | 0,602 | tuyến tính không đủ cho 8 lớp |
| Decision Tree | 0,892 | chỉ kém RF 0,019 mà header C++ nhỏ hơn ~100 lần |
| **Random Forest** | **0,911** | đang dùng |

---

## 3. Model đang nằm trong game: điểm mạnh / điểm yếu theo TỪNG lớp

Model đang tích hợp được train trên **6 trận người chơi thật** (11.562 frame, 366 hành
động) — không phải bộ 150 trận ở mục 2. Lý do chọn bộ nhỏ này: xem mục 5.

### 3.1 Target `action`

| Lớp | Số mẫu train | F1 | Đánh giá |
|---|---|---|---|
| None (không làm gì) | 11.196 | 0,756 | ổn |
| **Serve** | 27 | **0,746** | **lớp hành động tốt nhất** |
| Set | 104 | 0,383 | trung bình |
| SpikeMedium | 56 | 0,143 | yếu |
| SpikeStrong | 73 | 0,129 | yếu |
| Jump | 87 | 0,055 | rất yếu |
| SpikeLight | 14 | 0,021 | rất yếu |
| Spike | 5 | 0,000 | không học được |
| **Bump** | **0** | — | **không có dữ liệu** |

macro-F1 = **0,356** (baseline 0,151)

### 3.2 Target `move`

| Lớp | Số mẫu | F1 |
|---|---|---|
| Đứng yên | 10.128 | 0,925 |
| Sang phải | 983 | 0,569 |
| Sang trái | 451 | 0,286 |

macro-F1 = **0,580** (baseline 0,309)

---

## 4. Kết quả thực nghiệm TRONG GAME

Cùng một người chơi, cùng một buổi, cùng luật game. Bot ở sân phải.

| | **Bot luật** (6 trận) | **Bot ML** (5 trận) |
|---|---|---|
| Tổng frame | 29.294 | 26.814 |
| Tỉ số (người – bot) | 20 – 70 | 13 – 67 |
| Bot thắng % điểm | 78% | **84%** |
| Tổng số hành động của bot | 105 | **562** (×5,4) |
| — Serve | 70 | 140 |
| — Spike (cả 3 mức) | 35 | **419** (×12) |
| — Set | 0 | 2 |
| — Jump | 0 | 1 |
| — Bump | 0 | 0 |
| Bot đứng yên (% frame) | 72% | 71% |

### Đọc bảng này thế nào

**Tốt hơn:** bot ML **chủ động hơn gấp 5,4 lần** và thắng nhiều điểm hơn (84% so với 78%).

**Kém hơn / chưa tới:**

- **Di chuyển thì KHÔNG khác bot luật.** Đứng yên 71% so với 72% — gần như trùng khớp.
  Điểm `move` 0,580 có được là nhờ lớp "đứng yên" (F1 0,925); còn chọn hướng đi thì yếu
  (sang trái 0,286). Chưa đủ căn cứ để nói "bot ML di chuyển tốt".
- **Spike nhiều nhưng một chiều.** 419 cú đập nhưng **97% là `SpikeMedium`**. Nguyên nhân
  nằm ở bước tích hợp: đang lấy `argmax` (luôn chọn lớp xác suất cao nhất) nên model không
  bao giờ đổi cú. Các cú đó hợp lệ là nhờ **lớp luật chặn cú bay ra ngoài biên**, không
  phải nhờ model — đo được model đề xuất cú không hợp lệ 95% số lần ở sân phải.
- **Serve thì ngược với dự đoán ban đầu:** đây lại là lớp **mạnh nhất** (F1 0,746), bot
  giao bóng 140 lần bình thường.
- **Jump / Set / Bump gần như không dùng:** 1 / 2 / 0 lần trong 5 trận.

**Giới hạn của phép đo này:** 5–6 trận, một người chơi duy nhất, có thể có hiệu ứng người
chơi quen dần. Chưa đủ để kết luận bot ML mạnh hơn bot luật một cách chắc chắn. Phép đo
chặt hơn (bot ML vs bot luật tự đánh nhiều trận) đã có sẵn chế độ bot-vs-bot nhưng chưa chạy.

---

## 5. Phát hiện quan trọng nhất của giai đoạn này

> **Điểm offline cao KHÔNG bảo đảm bot chơi được trong game.**

Bằng chứng đo được:

| Bộ dữ liệu train | macro-F1 `action` offline | Hành vi trong game |
|---|---|---|
| 150 trận | **0,892** | Đề xuất `None` ở **8.014/8.014 frame** — bot bất động |
| 6 trận người chơi thật | **0,356** | Bot **thực sự hành động** 562 lần, thắng 84% điểm |

Nguyên nhân: **lệch phân bố trạng thái** (distribution shift).

| Feature | Trung bình ở tập train (150 trận) | Trung bình ở game thật |
|---|---|---|
| `self_dist_to_net` | 742 | 394 |
| `ball_y` | 414 | 549 |
| `can_act` | 0,55 | 0,95 |

Model gặp những tổ hợp **chưa từng có trong tập train** nên rơi vào nhánh `None`.
Ví dụ cụ thể đã gây lỗi: lúc chờ giao bóng, feature là `ball_alive = 1` nhưng
`ball_a = 0`, `ball_vx = 0` — bóng vừa "sống" vừa đứng im, tổ hợp này không tồn tại
trong 124.172 frame dữ liệu train. `Serve` có F1 = 1,00 trong cross-validation mà vẫn
vô dụng ở đây: trận đứng im 10.597 frame cho tới khi thêm luật chặn.

⇒ **Ràng buộc thật không phải là số lượng mẫu, mà là phân bố.** Dữ liệu phải được thu
*đúng ở vùng trạng thái mà bot sẽ gặp khi nó chơi*.

---

## 6. Vì sao bot còn một chiều — đã chẩn đoán xong

Không phải vì thiếu mẫu:

| Lớp | Số mẫu train | Số lần dùng trong 5 trận |
|---|---|---|
| Set | **104** (nhiều nhất) | **2** |
| SpikeLight | 14 | 6 |

`Set` có nhiều mẫu nhất mà gần như không được dùng. Lý do: `Set` (`ball_y` = 369) và
`SpikeLight` (`ball_y` = 376) **trùng nhau gần hoàn toàn** trong không gian feature —
model không tách được hai lớp này nên mặc định chọn lớp đông hơn.

Ba nguyên nhân, xếp theo mức ảnh hưởng:

1. **`argmax` ở bước tích hợp** — luôn chọn đúng 1 lớp ⇒ bot không bao giờ đa dạng.
   *Không cần dữ liệu mới để sửa.*
2. **Feature chưa tách được `Set` với `SpikeLight`** — cần feature mô tả "bóng đang lên
   hay đang xuống", "còn mấy lần chạm".
3. **`Bump` = 0 mẫu** — lỗ hổng dữ liệu thật sự duy nhất (người chơi chưa dùng phím Shift).

---

## 7. Việc tiếp theo

| Ưu tiên | Việc | Cần gì |
|---|---|---|
| 1 | Thay `argmax` bằng **lấy mẫu theo xác suất** (`predict_proba` rồi rút thăm) | ~20 dòng code, không cần dữ liệu mới |
| 2 | Thu dữ liệu `Bump` (chơi thêm ~5 trận, nhớ bấm Shift) | người chơi |
| 3 | **DAgger vòng 1**: thêm 5 trận vừa chơi với bot ML vào tập train | đã có 26.814 frame |
| 4 | Thêm feature tách `Set` / `SpikeLight` (bóng lên–xuống, số lần chạm còn lại) | code |
| 5 | Đo tự động ML-bot vs rule-bot nhiều trận (chế độ bot-vs-bot đã làm xong) | chạy máy |

**DAgger (Dataset Aggregation)** là cách xử lý đúng bài cho vấn đề ở mục 5: cho bot tự
chơi → thu lại chính những trạng thái bot tự đẩy mình vào → gán nhãn cho các trạng thái
đó → gộp vào tập train → train lại. Mỗi vòng, tập dữ liệu phình ra đúng hướng model đang
yếu. 5 trận vừa chơi chính là dữ liệu của vòng 1.

---

## 8. Phụ lục — thiết lập thí nghiệm

- **Chia dữ liệu:** GroupKFold theo `match_id` (hash SHA-256 + salt cố định) ⇒ không có
  frame cùng một trận nằm ở cả train và test. `salt` không đổi sau khi đã đo.
- **Chỉ số:** macro-F1 trên các lớp *có mặt*, ngưỡng tối thiểu 10 mẫu/lớp.
- **Lấy mẫu cửa sổ:** ±20 frame quanh thời điểm chạm bóng — giữ 124.172/483.329 frame
  (25,7%) nhưng **giữ 100% mẫu hành động**. Đã kiểm tra không thiên lệch: chỉ 8 lần hành
  động nằm ngoài cửa sổ, trên 359.157 frame bị loại.
- **Seed:** 42. Toàn bộ số liệu lưu trong `reports/experiments.md` (41 dòng thí nghiệm).

### Kết quả âm (negative result) đã ghi nhận

- **ADR-007 — Feature vận tốc không giúp gì.** Thêm `self_vx`, `self_vy`, `opp_vx`,
  `self_dx_10`: 0,485 → 0,483 (dữ liệu người chơi), 0,570 → 0,568 (dữ liệu bot).
  Nguyên nhân: `self_vx` và `self_vy` **bằng 0 ở 100% frame đập bóng** vì nhân vật dừng
  lại rồi mới đập. Đã hoàn lại.
- **ADR-008 — Dữ liệu tổng hợp để kiểm tra pipeline.** Trên dữ liệu sinh có quy luật biết
  trước, model đạt 0,917 trong khi trần lý thuyết là 0,92 ⇒ chứng minh **pipeline đúng**,
  mọi thất bại trên dữ liệu thật là do dữ liệu, không phải do bug code.
