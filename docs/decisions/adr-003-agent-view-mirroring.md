# ADR-003: Chuẩn hoá góc nhìn bằng cách lật sân qua lưới

- Trạng thái: Accepted
- Ngày: 2026-09-21

## Bối cảnh

Sân có hai phía. Player đứng bên trái (entity 0) và đánh sang phải. Bot đứng bên phải (entity 3) và đánh sang trái.
Nếu dùng toạ độ gốc thì:
- Model học từ người chơi (bên trái) không dùng được cho bot (bên phải), vì mọi toạ độ và hướng đều ngược.
- Dữ liệu hai phía không gộp chung để học được. Trong khi dữ liệu người chơi vốn đã ít.
- Nhãn `MoveLeft`/`MoveRight` mang nghĩa ngược nhau tuỳ phía: với bên trái là lùi/tiến, với bên phải là tiến/lùi.

## Các lựa chọn đã xét

| Lựa chọn | Nhận xét |
|---|---|
| Hai model riêng cho hai phía | Chia đôi lượng dữ liệu; phải bảo trì gấp đôi |
| Thêm feature `side` (0/1) | Model phải tự học sự đối xứng, tốn dữ liệu, dễ học sai |
| **Lật sân về một góc nhìn chung** | Tận dụng tính đối xứng có sẵn của bài toán. Phép biến đổi rất đơn giản, viết lại trong C++ dễ dàng |

## Quyết định

Mọi feature được tính trong **hệ toạ độ của agent**: "mình luôn ở bên trái, đánh sang phải".
Với agent bên phải:

```text
x'          = 2 * NET_X - x        (vị trí mình, đối thủ, bóng, landingX, đỉnh parabol)
ball_vx'    = -ball_speed          (đại lượng có hướng đổi dấu)
move'       = -move                (nhãn: +1 luôn là tiến về lưới)
```

`NET_X = 1370` lấy từ `BigRect::NET_X = offsetX + RECT_WIDTH / 2` của game, ghi trong `schema/*.json` mục `constants`.

Khi chạy trong game, `MLInputSystem` của bot bên phải phải lật toạ độ **trước** khi tính feature,
và đổi dấu `move` **sau** khi model dự đoán (`moveX = -move * SPEED`). Xem `docs/integration.md`.

## Hệ quả

- Một model dùng được cho cả hai phía. Dữ liệu người chơi (trái) và bot (phải) gộp chung được.
- Đây là một dạng *data augmentation* dựa trên đối xứng. Có thể nêu trong báo cáo.
- Rủi ro: quên lật ở một chỗ (bên Python hoặc bên C++) thì model vẫn chạy nhưng sai mà không báo lỗi.
  Các biện pháp giảm rủi ro:
  - test `test_v0_mirror_is_symmetric`: hai nhân vật đứng đối xứng phải cho feature giống hệt nhau;
  - test `test_v1_opponent_view` kiểm tra từng feature đã lật;
  - golden test phía C++.
- Nếu sau này sân không còn đối xứng (ví dụ gió, nửa sân khác nhau) thì phải xem lại ADR này.
