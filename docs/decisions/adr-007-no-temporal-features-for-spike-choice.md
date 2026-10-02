# ADR-007: Không dùng feature vận tốc cho `spike_choice` trên schema v0

- Trạng thái: Accepted
- Ngày: 2026-10-02

## Bối cảnh

Schema v0 không có cột bóng, nên 6 feature hiện tại chỉ là vị trí tĩnh của hai nhân vật tại
một frame. Mà quyết định đập cú nào phụ thuộc vị trí và quỹ đạo bóng ([ADR-006](adr-006-conditional-spike-choice-target.md)).

Giả thuyết: **hiệu vị trí giữa các frame là proxy cho hướng bóng**. Nhân vật chạy tới đón bóng,
nên vận tốc ngay trước khi đập cho biết bóng bay từ đâu tới — thông tin mà vị trí tĩnh không có.
Phép tính chỉ là phép trừ nên viết lại được trong C++ (luật 3 của `AGENTS.md`).

Đã thử 4 feature:

| Feature | Công thức |
|---|---|
| `self_vx` | `self_x(t) − self_x(t−1)` |
| `self_vy` | `self_y(t) − self_y(t−1)` |
| `opp_vx` | `opp_x(t) − opp_x(t−1)` |
| `self_dx_10` | `self_x(t) − self_x(t−10)` |

## Kết quả đo

macro-F1, Random Forest, cross-validation 5 fold theo trận:

| Dữ liệu | 6 feature | + 4 feature vận tốc |
|---|---|---|
| Người chơi (41 mẫu) | 0.485 ± 0.168 | **0.483 ± 0.172** |
| Bot (353 mẫu) | 0.570 ± 0.043 | **0.568 ± 0.040** |

Không cải thiện trên cả hai tập. Nguyên nhân nằm ở phân bố giá trị **tại đúng những frame mà
`spike_choice` được định nghĩa** (353 frame đập bóng của bot):

| Feature | Tỉ lệ bằng 0 | Feature importance |
|---|---|---|
| `self_vx` | **100,0%** | 0.000 |
| `self_vy` | **100,0%** | 0.000 |
| `self_dx_10` | 95,5% | 0.005 |
| `opp_vx` | 50,1% | 0.029 |

`AIInputSystem` chạy tới điểm rơi, **đặt `intent.moveX = 0`, rồi mới đập** ở frame sau. Nên tại
frame đập bóng nhân vật đang đứng im và mọi hiệu vị trí của chính nó bằng 0. `self_dx_10` cũng
bằng 0 vì nhân vật đã đứng chờ bóng rơi hơn 10 frame.

## Quyết định

**Không thêm feature vận tốc vào schema v0.** Đã hoàn lại toàn bộ thay đổi.

Lý do: chúng bằng 0 một cách hệ thống đúng ở những frame cần dùng, nên không mang thông tin;
đổi lại phải trả giá bằng 4 cột thừa, nguy cơ overfit trên 41 mẫu, và phía C++ phải nhớ lại
10 frame gần nhất.

## Hệ quả

- Hạn chế "v0 thiếu thông tin bóng" **không vá được bằng feature engineering**. Chỉ log v1 có
  cột bóng mới giải quyết được — xem `docs/data_contract.md`.
- Nếu sau này muốn thử lại feature theo thời gian, phải lấy trạng thái ở **vài frame TRƯỚC lúc
  đập** (ví dụ `self_x(t−15)`, lúc nhân vật còn đang chạy) chứ không phải hiệu tại frame đập.
  Nhưng khi đã có log v1 thì `ball_vx` và `ball_vertex_dx` nói trực tiếp điều đó, nên hướng này
  gần như không còn cần thiết.
- Với target `move` thì vận tốc **có** giá trị khác 0, nhưng dùng nó là rủi ro: model sẽ học
  "tiếp tục làm như frame trước" thay vì học hành vi, một cái bẫy quen thuộc của behavior
  cloning. Nếu thử thì phải đo riêng và so với mốc không có nó.
