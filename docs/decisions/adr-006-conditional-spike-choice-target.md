# ADR-006: Tách việc chọn cú đập thành nhãn có điều kiện `spike_choice`

- Trạng thái: Accepted
- Ngày: 2026-10-02

## Bối cảnh

Mục tiêu: bot học chọn một trong ba cú đập `SpikeLight` / `SpikeMedium` / `SpikeStrong`.

Nhãn `action` hiện có gộp mọi hành động vào một bài toán phân loại trên **mọi frame**. Đo trên
101 trận log v0 cho thấy cách gộp này không dùng được cho mục tiêu trên:

- 3 cú đập chiếm **353 / 241.542 frame = 0,15%**. Lớp `None` chiếm trên 99%.
- Hệ quả: `action` đạt macro-F1 0.258 ± 0.137, **đúng bằng majority** — độ lệch chuẩn còn lớn
  hơn khoảng cách tới majority, tức là hoàn toàn nhiễu.
- Trộn hai câu hỏi khác bản chất vào một model: *"frame này có nên đánh không"* (phụ thuộc thời
  điểm, cooldown, lượt chạm) và *"đã đánh thì đánh cú nào"* (phụ thuộc vị trí và quỹ đạo bóng).

Phía game, `AIInputSystem::update()` **đã** có luật quyết định khi nào tấn công:
`trend == TowardBotCourt && ball.y < AI_ATTACK_HEIGHT_THRESHOLD`. Việc duy nhất còn lại cho model
là chọn cú — đúng chỗ ghi đè `intent.finalIntent` ngay trước `intentPool.add`.

## Các lựa chọn đã xét

| Lựa chọn | Nhận xét |
|---|---|
| Giữ một model `action` cho mọi frame | Mất cân bằng 1:650. Đã đo: bằng majority, vô dụng |
| Lấy mẫu lại (undersample lớp `None`) | Vẫn trộn hai câu hỏi; tỉ lệ lấy mẫu thành một hyperparameter phải tune, khó giải thích trong báo cáo |
| Cửa sổ frame quanh lúc chạm bóng | Giảm mất cân bằng nhưng không hết; vẫn cần chọn độ rộng cửa sổ |
| **Nhãn có điều kiện `spike_choice`** | Bài toán 3 lớp tương đối cân bằng, khớp đúng chỗ ghép vào game, không có hyperparameter lấy mẫu nào phải biện minh |

## Quyết định

Thêm nhãn `spike_choice` vào `schema/*.json` với `conditional: true`:

- `features.build()` đặt `spike_choice` = tên cú đập ở frame agent thật sự đập, **NaN** ở frame khác.
- `train.py` loại các dòng NaN khỏi train / dev / CV / holdout của riêng target đó. Các target khác
  trong cùng run (vd `move`) vẫn dùng toàn bộ frame.
- Không có mẫu nào thì báo lỗi rõ ràng, trỏ sang `scripts/count_spikes.py`, thay vì để sklearn
  ném traceback.

Cơ chế `conditional` là tổng quát, không gắn riêng với spike: nhãn nào chỉ xác định trên một phần
frame đều khai báo được như vậy.

## Hệ quả

- **Phát hiện ngoài dự đoán:** trên log v0 — *không có cột bóng nào* — Random Forest vẫn đạt
  **0.570 ± 0.043** so với majority 0.236. Nguyên nhân: bot luật luôn chạy tới đúng điểm rơi rồi
  mới đánh, nên `self_x` của chính nó là **proxy cho vị trí bóng** (feature importance 0.29, cộng
  thêm `dx_opp` 0.29 và `self_dist_to_net` 0.25 đều dẫn xuất từ `self_x`). `PickAttackIntent` lọc
  các cú rơi trong sân đối phương trước khi random, và tập hợp lệ đó phụ thuộc khoảng cách tới lưới:

  | Cú đập | `self_x` trung bình (hệ đã lật) |
  |---|---|
  | SpikeLight | 1189.9 — gần lưới |
  | SpikeMedium | 987.8 |
  | SpikeStrong | 923.0 — xa lưới |

  Nhận định trước đó rằng *"học từ bot là học `std::rand()`, không học được gì"* là **sai**: phần
  random chỉ áp dụng **trong** tập cú hợp lệ, còn tập hợp lệ thì tất định và học được.

- **Nhưng kết quả này không suy ra được cho người chơi**, vì hai lý do:
  1. Người thật không đứng đúng điểm rơi như bot, nên `self_x` là proxy yếu hơn nhiều.
  2. `std::rand()` trong tập hợp lệ **chặn cứng trần hiệu năng** — không model nào vượt được.
     Muốn bỏ trần này thì bên game phải chọn cú tất định.

- Để đạt mục tiêu thật (bắt chước **người**) vẫn cần đủ ba điều kiện, chưa cái nào xong:
  1. Log v1 có cột bóng (`docs/data_contract.md`) — hiện người chơi chỉ có 41 mẫu và v0 không có bóng
  2. Bỏ `std::rand()` trong `AIInputSystem::PickAttackIntent`
  3. ≥ 150 mẫu mỗi cú từ người chơi — hiện 17 / 19 / 5. Kiểm tra bằng `scripts/count_spikes.py`

- Rủi ro của nhãn có điều kiện: model **không biết khi nào nên đánh**, nên trong game bắt buộc phải
  có luật quyết định thời điểm. Nếu sau này muốn model tự quyết cả thời điểm thì cần một nhãn riêng
  (vd `should_attack` dạng nhị phân) chứ không mở rộng `spike_choice`.
