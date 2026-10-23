# BÁO CÁO TIẾN ĐỘ 3: VÒNG TỰ HỌC VỚI CHUYÊN GIA VÀ MODEL ĐANG DÙNG TRONG GAME

> Thời gian: 16/10/2026 – 23/10/2026. Mọi con số dưới đây đo bằng trận **bot luật (trái) vs bot ML (phải)**
> chạy không cửa sổ (MatchRunner), cùng seed thì lặp lại y hệt. Chi tiết từng thí nghiệm:
> `ProjectTotNghiep-AI/reports/experiments.md`, lý do quyết định: `docs/decisions/` (ADR-011 – ADR-015).

---

## 📌 Tóm tắt

1. **Model đang dùng trong game chính (v10 vòng 2)**, đấu bot luật 100 trận:
   - Thắng 98/100 trận, ghi **80,3%** số điểm (1381 – 338), bot ML chỉ mất 3,4 điểm/trận.
   - Lối chơi đa dạng: Set → Jump → Spike, đập ngay khi bóng vào vùng detect (~62 lần/trận), Bump cứu bóng.
2. **Cách học:** Behavior Cloning kết hợp **DAgger tự động**: bot tự chơi, một nửa số lượt do **chuyên gia
   (quy tắc ghi điểm viết tay)** điều khiển, model học lại từ đó; model mới chỉ được giữ khi đo tốt hơn.
3. **Phát hiện quan trọng nhất:** để **model tự quyết có nhảy** sau Set (vật lý chỉ chọn frame nhảy) thay cho
   rút thăm 31,4%: % điểm **72,6% → 93,8%**, số pha Set → Jump → Spike ghi điểm **gần gấp đôi** – không cần train lại.
4. **Trung thực về giới hạn:** kỹ năng ghi điểm chủ yếu học từ chuyên gia quy tắc, chưa phải từ người chơi;
   **chưa đánh giá với người chơi thật**.

---

## 1. Việc đã làm từ báo cáo 2

| #   | Việc                                                                                | Kết quả / bằng chứng                                                                                                                  |
| --- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Phát hiện **rò rỉ nhãn** do thứ tự ghi log (game ghi log SAU khi áp hành động)      | Ghép nhãn lùi 1 frame. Macro-F1 thật là **0,438** (bản rò rỉ báo 0,782) – ADR-011                                                     |
| 2   | Đo offline **đúng cách bot ra quyết định trong game** (ngưỡng, rút thăm, luật chặn) | `eval_policy`, ADR-013                                                                                                                |
| 3   | Phát hiện **dữ liệu sinh tự động** trong `Logs/v1`                                  | 298/338 file là dữ liệu tổng hợp; dữ liệu người thật: **30 trận**                                                                     |
| 4   | **MatchRunner**: bot vs bot không cửa sổ, ~1.000× thời gian thật, theo seed         | 100 trận chạy ~5 phút (≈ 25 giờ chơi thật)                                                                                            |
| 5   | **Vòng tự học tự động**: tự chơi → train → export C++ → build → đo → giữ/bỏ         | `python -m spike_ai.selfplay` – ADR-014                                                                                               |
| 6   | Thử **self-imitation** (bot học pha chính nó thắng)                                 | **Kết quả âm**: không thắng nhiều hơn                                                                                                 |
| 7   | **Chuyên gia ghi điểm** làm người dạy (DAgger tự động)                              | Chuyên gia 79% → ~97% điểm sau khi sửa lỗi; bot học theo +3–4 điểm % – ADR-015                                                        |
| 8   | Phát hiện **thiên lệch khi chọn model** trên cùng bộ đo                             | 51,0% trên bộ dùng để chọn → 49,7% trên bộ mới. Sửa: **xác nhận 2 bước** trên 2 bộ seed khác nhau, giữ 1 bộ **để dành** (seed 400000) |
| 9   | **Sửa luật chặn cú đập** của bot ML (tính bước bay cuối + qua lưới)                 | Cú đập lưới 704 → 2 điểm; trận thắng **65 → 100/100**                                                                                 |
| 10  | Phân tích **chuỗi ghi điểm**                                                        | Chỉ **SpikeStrong trên không** ghi điểm, và chỉ khi Set cách lưới ≥ 430 px                                                            |
| 11  | **Model tự quyết nhảy** (vật lý chọn frame)                                         | 72,6% → **93,8%** điểm                                                                                                                |
| 12  | **Ép học đập ngay** (lối chơi đa dạng) – model v10                                  | Bot đập ngay ~62 lần/trận, vẫn thắng 100/100 trận                                                                                     |
| 13  | Đưa model vào game chính, test tự động                                              | Commit riêng, revert được; **163 test** Python pass                                                                                   |

---

## 2. Tiến triển của bot ML qua các mốc

Đo trên bộ trận **để dành** (seed 400000, không dùng để chọn model), 50 trận:

| Mốc                       | Model                                    | Cách nhảy sau Set | % điểm thắng                      |
| ------------------------- | ---------------------------------------- | ----------------- | --------------------------------- |
| Đầu tuần                  | DAgger với người (`dagger_v1_w_aligned`) | rút thăm 31,4%    | ~46% (đo trên seed 100000/200000) |
| Sau tự học v1             | học từ chuyên gia                        | rút thăm 31,4%    | 48,3%                             |
| Sau tự học v3             | xác nhận 2 bước                          | rút thăm 31,4%    | 50,1%                             |
| + sửa luật chặn cú đập    | v3                                       | rút thăm 31,4%    | 72,6%                             |
| + model tự quyết nhảy     | v3                                       | **model quyết**   | **93,8%**                         |
| **Đang dùng: v10 vòng 2** | ép học đập ngay                          | model quyết       | 82–83%                            |

**Nhận xét:** bước tăng lớn nhất không đến từ train thêm mà từ **sửa đúng nút thắt** (luật chặn, luật nhảy) sau
khi phân tích điểm mất ở đâu.

---

## 3. Model đang dùng trong game (v10 vòng 2) – 100 trận

| Chỉ số                     | Bot luật  | Bot ML       |
| -------------------------- | --------- | ------------ |
| Trận thắng                 | 2         | 98           |
| % điểm thắng               | 19,7%     | **80,3%**    |
| Điểm mất / trận            | 13,8      | **3,4**      |
| Set / Bump / Jump mỗi trận | 0 / 0 / 0 | 72 / 10 / 64 |
| Chuỗi Set → Jump → Spike   | 0         | 6.317        |
| Đập ngay quả đầu / trận    | –         | ~62          |
| Jump bừa (không sau Set)   | 0         | **0**        |

**Bot ML ghi 1381 điểm bằng:** SpikeStrong sau Set → Jump **766 (55%)** · bot luật tự đánh hỏng 358 (26%) ·
giao bóng ăn điểm 126 (9%) · đập ngay 18.

**Bot ML mất 338 điểm vì:** Bump hỏng 139 · không đỡ được SpikeMedium 83 · giao bóng hỏng 59 · Set hỏng 34.

**Vì sao chọn v10 vòng 2 chứ không phải bản thắng nhiều nhất (v3 tự quyết nhảy, ~92–94%):** v3 gần như không bao
giờ đập ngay; v10 vòng 2 **chơi đa dạng hơn, giống người hơn**, vẫn thắng mọi trận. Đánh đổi −12 điểm %.

| Vòng v10 (seed 400000) | % điểm    | Đập ngay (50 trận) |
| ---------------------- | --------- | ------------------ |
| v3 (đối chứng)         | 93,8%     | 15                 |
| vòng 1                 | 83,0%     | 2.028              |
| **vòng 2 (chọn)**      | **82,0%** | **2.995**          |
| vòng 3                 | 71,0%     | 4.399              |
| vòng 5                 | 74,4%     | 4.379              |

---

## 4. Các kết quả âm (có số liệu, đưa vào báo cáo)

| Thử nghiệm                                        | Kết quả                       | Bài học                                                                                  |
| ------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------- |
| Self-imitation                                    | Không thắng nhiều hơn         | Bot chép chính nó thì không giỏi hơn chính nó                                            |
| Train liên tục 2 giờ (24 vòng)                    | 0 vòng được giữ               | 85% điểm mất do **luật chặn**, không phải do model – phải phân tích trước khi train thêm |
| Chuyên gia đập ngay để ghi điểm                   | 0/486 cú ghi điểm             | Bot luật chạy tới điểm rơi tức thì nên luôn đỡ được cú đập thấp                          |
| v6 / v8: dạy đập từ đất, quy ước theo khoảng cách | 45–85%, kém dần qua từng vòng | Dạy hành vi không ghi điểm làm model yếu đi                                              |
| v9: chỉ học pha ghi điểm + đập ngay               | Model không học được đập ngay | Pha đập ngay hiếm khi ghi điểm nên bị lọc gần hết                                        |

---

## 5. Cách bot ML ra quyết định (để trả lời câu "bot học ở đâu?")

| Quyết định                                   | Ai quyết                              | Học từ đâu                                                                 |
| -------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------- |
| Đánh gì (Set / Bump / cú đập nào / đập ngay) | **Model** (Random Forest, 21 feature) | Người chơi (30 trận thật + dữ liệu tổng hợp) + chuyên gia quy tắc (DAgger) |
| Có nhảy sau Set không                        | **Model**                             | Chuyên gia                                                                 |
| Nhảy đúng frame nào                          | Vật lý (mô phỏng quỹ đạo)             | – (model không học được timing, ADR-012)                                   |
| Cú đập có hợp lệ không                       | Luật chặn an toàn                     | – (chỉ chặn cú ra ngoài / đập lưới)                                        |
| Di chuyển                                    | Luật gốc                              | –                                                                          |

---

## 6. Tiến độ tổng thể (~80–85%)

| Giai đoạn                                                   | Tình trạng                                                                |
| ----------------------------------------------------------- | ------------------------------------------------------------------------- |
| GĐ0–2: pipeline, schema v1, logger, export C++, golden test | ✅ Xong                                                                   |
| GĐ3: dữ liệu người chơi                                     | ⚠️ Một phần (30 trận thật, mục tiêu ≥ 8 người × 5 trận)                   |
| GĐ4–5: train, chọn model, tích hợp vào game                 | ✅ Xong (v10 vòng 2 trong game chính)                                     |
| GĐ6: đánh giá trong game                                    | ⚠️ Bot vs bot xong; **chưa** đo người vs bot ML, độ giống người, khảo sát |
| GĐ7: mở rộng (DAgger, tự học)                               | ✅ Phần lớn; ❌ báo cáo, slide, video demo chưa làm                       |

---

## 7. Kế hoạch tiếp theo

1. **Cho người thật chơi với bot ML** (≥ 20 trận, ưu tiên người chưa từng chơi → holdout): tỉ lệ thắng, cách ghi điểm.
2. **So sánh độ giống người**: phân bố hành động, vị trí đứng, thời điểm chạm bóng – bot ML vs người vs bot luật.
3. **Khảo sát** (Likert 1–5): "bot có chơi giống người không?", có thể làm blind test.
4. Viết báo cáo, slide, quay video demo.

## 8. Hướng phát triển mở rộng

1. Gom log nhiều người chơi, **định kỳ train lại** và chỉ cập nhật khi đo tốt hơn (giống vòng tự học hiện tại).
2. **Bot riêng theo từng người chơi** (train theo `player_id`), phân cụm phong cách chơi.
3. **Độ khó thích ứng** theo trình độ người chơi.
4. **Kết hợp Reinforcement Learning** sau khi Behavior Cloning cho điểm xuất phát, dùng MatchRunner (~1.000× thời gian thật).
5. **Model chuỗi** (MLP/GRU qua ONNX) để tự học cả thời điểm nhảy, bớt phụ thuộc luật vật lý.
