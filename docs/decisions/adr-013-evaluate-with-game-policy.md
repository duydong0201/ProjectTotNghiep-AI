# ADR-013: Đo model bằng đúng cách bot ra quyết định trong game, trên mọi frame

- **Ngày:** 19/10/2026
- **Trạng thái:** Accepted
- **Liên quan:** ADR-004 (cách chia dữ liệu), ADR-011 (rò rỉ nhãn), ADR-012 (bot lai model + vật lý)

## Bối cảnh

`train.py` chấm điểm bằng **argmax** trên các frame **±20 quanh lúc chạm bóng**. Game làm khác hẳn:
`AIInputSystem.cpp` hỏi model ở **mọi frame**, ra tay khi `1 − P(None) > 0,3`, rút thăm có
temperature, rồi qua các luật chặn (vùng detect, lượt giao, không tự nhảy) và luật nhảy sau Set.
Vì thế macro-F1 của `train.py` không nói gì về con bot thật. Thí dụ: với model không rò rỉ nhãn,
argmax gần như luôn ra `None`, trong khi bot vẫn đánh được nhờ ngưỡng.

## Quyết định

- `src/spike_ai/policy.py` chép cách ra quyết định của game sang Python. Hằng số ở `GAME_POLICY`
  phải khớp `AIInputSystem.cpp`.
- Tính **phân phối** hành động của bot ở mỗi frame thay vì rút thăm. Chỉ số là kỳ vọng, không phụ
  thuộc seed.
- `python -m spike_ai.eval_policy --config <config>`: cross-validation theo trận như `train.py`,
  fit trên đúng các dòng `train.py` dùng, nhưng chấm trên **mọi frame** của trận để riêng. In ba
  cách đo cạnh nhau: argmax / cửa sổ, argmax / mọi frame, bot trong game.
- Ngoài F1 theo đúng frame, báo thêm **tỉ lệ bắt đúng cú trong ±3 frame** (hành động chỉ dài 1
  frame, đập sớm 1 frame vẫn là đúng lúc) và **số lần ra tay mỗi trận**.
- Model sắp đưa vào game phải đo bằng `eval_policy`, không chỉ bằng `train`.

Chưa chép (ghi rõ trong `policy.py`): phần luật gốc của bot, cú đập trên không sau khi bot tự nhảy,
lọc "cú đập phải rơi trong sân", và cooldown. Ba phần đầu cần vị trí CỦA BOT, mà dữ liệu offline
đi theo vị trí người chơi.

## Kiểm chứng công thức vùng detect

Trên 30 trận chơi thật, cú đập được game **chấp nhận** có khoảng cách tâm bóng – tâm nhân vật trung
vị 29–34% chiều cao (p90 < 50%), cú bị **bỏ qua** trung vị 74–82%. Công thức khớp với game.

## Kết quả đầu tiên và một phát hiện về dữ liệu

| | argmax / cửa sổ | argmax / mọi frame | bot trong game |
|---|---|---|---|
| macro-F1, 330 trận `Logs/v1` | 0,411 | 0,388 | **0,173** |
| macro-F1, chỉ 30 trận chơi thật | 0,108 | 0,110 | **0,141** |

Điểm tụt mạnh trên 330 trận là do **dữ liệu**, không phải lỗi đo: **298/338 file** trong `Logs/v1`
(các ngày 06, 10, 11/10, tên P01–P04) do script sinh ra (`make_synthetic_v1.py`,
`generate_from_player_profile.py`, `augment_real_logs.py`). Ở các file đó **100%** cú chạm xảy ra
khi bóng đúng ở `y = 305`, kể cả cú đập khi nhân vật đang ở độ cao ~520, tức bóng nằm dưới chân
hơn 200 px; game thật không bao giờ chấp nhận cú như vậy. Trên file chơi thật, tỉ lệ đó là **0%**,
kể cả các file cùng phiên bản game `67e1f38`. Luật vùng detect của game chặn đúng các cú "không thể"
này, nên F1 cú đập tụt từ ~0,77 xuống ~0,04.

Hệ quả: mọi model train trên toàn bộ `Logs/v1`, kể cả model đang chạy trong game, đã học phần lớn
từ dữ liệu có vật lý không khớp game. Các kết quả cũ trong `reports/experiments.md` và `WORKLOG.md`
trên các tập đó cần đọc với lưu ý này.

## Hệ quả

- Con số báo cáo cho "bot trong game" từ nay lấy từ `eval_policy`, kèm số trận thật.
- **Quyết định của người dùng (19/10/2026): KHÔNG tách** - tiếp tục train trên toàn bộ `Logs/v1`, gồm
  cả dữ liệu sinh tự động. Khi báo cáo, ghi rõ tập train gồm 30 trận chơi thật + 298 trận sinh tự động, và
  nêu con số `eval_policy` trên riêng 30 trận thật bên cạnh con số trên toàn bộ.
