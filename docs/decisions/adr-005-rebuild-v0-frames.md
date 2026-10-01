# ADR-005: Dựng lại frame cho log v0 trước khi tính feature

- Trạng thái: Accepted
- Ngày: 2026-09-25

## Bối cảnh

Log v0 của game **không ghi theo frame mà ghi theo sự kiện**. `ApplyHorizontalMovement`
(`Source/Utils/ApplyIntentToComponent.h`) duyệt `CharacterIntentPool` và ghi **một dòng cho mỗi
nhân vật đang có intent**; trong dòng đó nhân vật còn lại luôn bị ghi cứng thành `"None"`.
`PlayerInputSystem` lại chỉ thêm intent vào pool khi thật sự có phím được bấm.

Ba hệ quả làm hỏng nhãn `move`:

1. **Nhãn `None` giả.** Khi cả hai cùng di chuyển, một frame sinh ra hai dòng, mỗi dòng ghi sai
   nhân vật kia thành `None`. Ví dụ thật (`Logs/2026-08-06_15-46-15.csv`, frame 416):

   ```text
   416, 596,265, MoveRight, 1801,265, None       <- Bot đang MoveLeft
   416, 596,265, None,      1794,265, MoveLeft   <- Player đang MoveRight
   ```

   Đo trên toàn bộ log hiện có: **40.838 / 209.552 dòng (19,5%) mang nhãn sai**.

2. **Mất nhãn "đứng yên".** Frame không ai bấm phím thì không sinh ra dòng nào. Khoảng 45% số
   frame trong khoảng thời gian của trận hoàn toàn vắng mặt trong log.

3. **Vị trí lệch giữa các dòng cùng frame.** Dòng ghi sau đã phản ánh vị trí vừa cập nhật của
   nhân vật ghi trước (frame 416: `BotPosX` là 1801 ở dòng đầu, 1794 ở dòng sau).

Ngoài ra `hasEvent = false` khi vị trí bị `clampf`, nên frame "đang ép sát biên sân" cũng bị bỏ.

## Các lựa chọn đã xét

| Lựa chọn | Nhận xét |
|---|---|
| Sửa logger bên game rồi thu lại dữ liệu | Đúng nhất về lâu dài, nhưng vứt bỏ toàn bộ 79 trận đã có và phải chờ bên game. Đây chính là nội dung log v1 (GĐ2) |
| Bỏ qua, coi như nhiễu | Không được: 19,5% nhãn sai là sai hệ thống chứ không phải nhiễu ngẫu nhiên, model sẽ học "thỉnh thoảng đứng yên vô cớ" |
| **Dựng lại frame khi đọc log** | Thông tin chưa mất vì mỗi dòng đều có số frame. Gộp lại là khôi phục được |

## Quyết định

Thêm `data.rebuild_frames_v0()`, chạy tự động trong `load_match()` với mọi file v0:

1. Gộp các dòng cùng `Frame` thành một dòng: mỗi nhân vật lấy event khác `"None"` nếu có.
2. Vị trí lấy ở **dòng cuối cùng** của frame — lúc cả hai nhân vật đã di chuyển xong, nên
   trạng thái nhất quán (dòng sự kiện bóng do `LogTrajectoryEvent` ghi luôn đứng trước dòng
   di chuyển trong cùng frame).
3. Điền các frame trống bằng vị trí giữ nguyên và event `"None"` — đây là nhãn "đứng yên" thật.
   Chỉ điền lỗ hổng dài **tối đa `MAX_FILL_GAP = 60` frame** (1 giây ở 60 fps).

Ngưỡng 60 frame chọn theo phân bố lỗ hổng đo được: phần lớn lỗ hổng dài 6–30 frame (đứng yên
thật trong pha bóng), còn lỗ hổng trên 100 frame là lúc reset điểm — nhân vật bị đặt lại vị trí
nên điền vào sẽ tạo ra chuỗi "đứng yên" giả kèm một cú nhảy vị trí.

## Hệ quả

- Mọi số liệu đo trước ngày 25/09/2026 trong `reports/experiments.md` được tính trên nhãn hỏng,
  **không dùng để so sánh** với kết quả sau ADR này. Phải train lại từ đầu.
- Cách chia dữ liệu không bị ảnh hưởng: vẫn nhóm theo `match_id`, không đổi `split.salt`. Với v0 thì
  [ADR-004](adr-004-data-split-strategy.md) đã bỏ holdout và chuyển sang cross-validation, nên kết quả
  phải đo lại bằng `dev_mode: kfold` chứ không phải tập dev đơn lẻ.
- Việc này **không thay thế** log v1. v1 vẫn cần thiết vì thiếu thông tin bóng thì nhãn `action`
  vẫn không học được — xem `docs/data_contract.md`.
- Khi log v1 ra đời, logger ghi mỗi frame một dòng nên `rebuild_frames_v0` chỉ áp dụng cho v0.
- Rủi ro còn lại: vị trí trong log là **sau** khi intent đã được áp dụng, nên feature ở frame t
  thực ra là trạng thái ngay sau hành động t. Đây là lệch một frame có hệ thống, đã ghi lại để
  xử lý ở GĐ4 (mục "căn nhãn" trong `PLAN.md`).
