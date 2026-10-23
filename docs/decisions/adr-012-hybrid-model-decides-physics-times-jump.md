# ADR-012: Model quyết định đánh gì, vật lý quyết định lúc nhảy

- **Ngày:** 16/10/2026
- **Trạng thái:** Accepted - sửa 23/10/2026: **model quyết có nhảy**, vật lý chỉ chọn frame (thay rút thăm
  31,4%): 72,6% -> 93,8% điểm (ADR-015). Chưa đo với người chơi thật
- **Liên quan:** ADR-011 (sau khi bỏ rò rỉ nhãn mới lộ ra giới hạn này)

## Bối cảnh

Sau ADR-011, model `played_v1_aligned_20261016-112201` (Random Forest, 21 feature, nhãn
ghép lùi 1 frame) là model trung thực đầu tiên: macro-F1 0,438 (bản rò rỉ: 0,782).
Từng intent:

| intent | F1 |
|---|---|
| SpikeMedium | 0,770 |
| SpikeStrong | 0,738 |
| Set | 0,512 |
| SpikeLight | 0,494 |
| Bump | 0,165 |
| **Jump** | **0,000** |
| **Serve** | **0,000** |

Người chơi muốn bot đánh chuỗi: đối thủ đánh sang → Set → Jump (lên cao) → Spike
Light/Medium/Strong, "tuỳ quả" — lúc đập luôn từ mặt đất, lúc Set rồi nhảy đập.

## Vì sao model không học được Jump

Không phải thiếu feature. Đã thử thêm 2 feature "còn bao nhiêu frame nữa bóng tới"
(mô phỏng `x += speed`, `y = a(x+b)^2 + c`, sai số 0,07 px trên log): Jump F1 vẫn 0,000.

Nguyên nhân: ngay ở đúng khoảnh khắc thích hợp, người chơi chỉ nhảy **16,3%** số lần
(3.314 frame), còn lại đập từ mặt đất / đỡ bóng / để bóng đi. Model học từng frame thì
ở những frame giống nhau chỉ có thể đưa ra P(Jump) ≈ 0,16 — không bao giờ thắng `None`
khi chọn bằng argmax. Đây là giới hạn của **cách ra quyết định từng frame bằng argmax**
với một hành động **hiếm, ngẫu nhiên và cần đúng thời điểm**, không phải lỗi dữ liệu.

Cùng lý do, với model trung thực, argmax gần như luôn ra `None` cho mọi hành động: đo trên
trạng thái bot gặp, P(ra tay) > 0,3 ở 20,3% frame bóng trong tầm, chỉ 0,53% khi bóng xa —
model biết lúc nào nên đánh, nhưng hiếm khi > 0,5.

## Quyết định

Chia việc (trong `Source/System/AIInputSystem.cpp`):

1. **Có ra tay không — model, bằng ngưỡng:** ra tay khi `1 − P(None) > 0,3`
   (`USE_ACT_THRESHOLD`, `ACT_THRESHOLD`), thay cho argmax.
2. **Ra tay bằng gì — model:** rút thăm theo xác suất model, temperature 3, chỉ trong các
   cú hợp lệ (cú đập phải rơi trong sân). Model **không** được tự đề xuất Jump
   (`MODEL_MAY_PROPOSE_JUMP = false`) — nếu cho, nó sẽ nhảy ngẫu nhiên ~16% ở mọi frame tương tự.
3. **Có nhảy đập sau Set không — rút thăm theo dữ liệu:** sau cú Set của chính mình, người
   chơi nhảy 31,4% số lần (383/1.220 cú Set, 330 trận Logs/v1), 34,0% đập từ mặt đất.
   Bot rút thăm một lần cho mỗi cú Set với `P_JUMP_AFTER_SET = 0,314`.
4. **Lúc nào nhảy — vật lý:** mỗi frame, mô phỏng "nếu nhảy ngay bây giờ thì sau bao nhiêu
   frame bóng vào vùng đập (< 50% chiều cao nhân vật)", chép đúng cơ chế nhảy của game
   (+19,62 px frame đầu, +9,81 px/frame tới frame 30, rồi rơi 9,81 px/frame). Nhảy ở frame
   đầu tiên kết quả nằm trong [18, 25].
5. **Đập trên không — model chọn cú:** khi đã nhảy theo kế hoạch và bóng vào vùng, rút thăm
   trong 3 cú đập theo xác suất model; nếu không cú nào hợp lệ thì dùng cách chọn của bot luật.

Không nhảy ở các tình huống khác. Không nhảy khi giao bóng (người chơi giao bóng bật nhảy
chỉ 3,8% số lần giao) — đây là phần chưa làm.

## Kiểm chứng offline của luật nhảy

Áp đúng luật (bước 4) lên 207 chuỗi Set → Jump → Spike **đập trúng** của người chơi:

| cửa sổ | không kích hoạt | lệch frame nhảy (luật − người), trung vị | \|lệch\| ≤ 6 frame |
|---|---|---|---|
| [20, 31] | 28/207 | −10 | 30% |
| [20, 26] | 28/207 | −2 | 67% |
| **[18, 25]** | **28/207** | **0** | **74%** |
| [16, 22] | 28/207 | +3 | 75% |

Ở đúng frame người chơi nhảy, mô phỏng đoán thời điểm chạm bóng lệch thực tế trung vị 1 frame.

Phát hiện phụ: 150/357 chuỗi người chơi bấm đập sau khi nhảy nhưng bóng **không** ở trong
vùng (đập hụt, game bỏ qua). Số "12 frame từ Jump tới Spike" đo trước đó đến từ các lần
bấm hụt này (164 chuỗi có đúng 12 frame), không phải thời điểm đập thật — đã bỏ.

## Hệ quả và giới hạn (ghi vào báo cáo)

- Bot ML giờ là **lai**: phần "đánh gì" học từ người chơi; phần "lúc nào nhảy" là luật
  vật lý viết tay, chỉ có **tỉ lệ** nhảy (31,4%) và **cửa sổ** [18, 25] lấy từ dữ liệu.
  Phải nói rõ điều này khi báo cáo, không được trình bày như model tự học được chuỗi.
- Behavior cloning học từng frame, chọn bằng argmax, không học được hành động hiếm cần
  đúng thời điểm. Hướng làm đúng hơn (ngoài phạm vi): model ra quyết định theo sự kiện
  ("sau cú Set: nhảy hay không"), hoặc model chuỗi thời gian.
- Ngưỡng 0,3 và cửa sổ [18, 25] chưa được đo trong trận thật. Thay vào game chỉ được giữ
  nếu 4 chỉ số trong game tốt hơn bản cũ; bản cũ đã sao lưu ở
  `ProjectTotNghiep/ModelBackups/truoc_A_C_20261016/`.
