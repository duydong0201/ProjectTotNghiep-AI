# ADR-011: Rò rỉ nhãn do thứ tự ghi log — ghép nhãn lùi một frame

- **Ngày:** 16/10/2026
- **Trạng thái:** Đã áp dụng (tuỳ chọn `label_shift: 1` trong config)
- **Liên quan:** ADR-007, ADR-010 (các kết quả âm có thể cần đọc lại dưới ánh sáng lỗi này)

## Bối cảnh

Suốt dự án có một nghịch lý lặp lại: điểm offline cao nhưng bot trong game kém.

| | offline | trong game |
|---|---|---|
| model 150 trận | macro-F1 0,892 | `None` ở 8.014/8.014 frame |
| model 120 trận | 0,994 trên holdout | 0,157 trên trận chơi thật |
| DAgger ×20 | `Set` đúng 97–100% ở trận DAgger giữ riêng | 0 lần `Set` khi đấu với người |

Manh mối cuối cùng: tại 45/45 frame người chơi bấm `Set` trong buổi DAgger, `ball_a = -8`.
Đó là hệ số quỹ đạo của chính cú `Set` (`TrajectoryData(-8.0f, 3.0f)`).

## Nguyên nhân

Thứ tự trong một frame của game:

```
MainScene
  _generateIntent->update()             GenerateIntent.cpp
      1. hệ thống input / AI sinh ý định
      2. updateCharacterActionStateSystem  -> nhân vật vào cooldown
      3. jumpStartSystem                   -> bắt đầu nhảy
      4. updateTrajectory                  -> đánh bóng, cập nhật RallyState
  _datasetLogger->LogFrame()            <- ghi log Ở ĐÂY, tức SAU bước 2-4
  ApplyIntentToComponent()
```

Comment đặt cạnh lời gọi ghi log (viết ngày 09/10/2026) khẳng định *"state trong dòng log
là state lúc ra quyết định"*. Điều đó sai: `GenerateIntent::update()` đã áp cú đánh trước khi
trả về.

## Bằng chứng

Trên 2.871 lần bấm phím (120 trận người chơi), so dòng có nhãn (frame k) với dòng ngay trước:

| | frame k (dòng có nhãn) | frame k−1 |
|---|---|---|
| nhân vật rảnh tay | 0% (cooldown vừa nạp 150–350 ms) | **99,2–100%** |
| bóng đã mang quỹ đạo cú đánh | 99,7–100% | chưa |
| người chạm cuối là "mình" | 100% | 4–45% |

Kiểm lại trên 40 trận sau khi sửa, tại các dòng có hành động: `can_act = 1` từ **0% → 99,9%**,
`action_remain_ms` trung bình từ **295 ms → 0 ms**.

## Quyết định

Ghép nhãn ở frame k với trạng thái ở frame k−1, trong cùng trận (`label_shift: 1`,
`train.py::build_rows`). Ai ra quyết định (cột `controller`, dùng cho lọc người / DAgger) cũng
lấy ở frame k, vì đó là frame quyết định được đưa ra.

Không sửa thứ tự ghi log trong game. Lý do: giữ một định dạng duy nhất cho toàn bộ hơn 330
trận đã thu, không phải đánh dấu "log trước / log sau" cho từng file. Trạng thái ở frame k−1
lệch với trạng thái thật lúc quyết định đúng một bước vật lý (~16 ms), và gần với trạng thái
bot nhìn thấy lúc chạy trong game hơn cách cũ (`AIInputSystem::update` chạy ở bước 1, trước
khi áp bất cứ thứ gì).

Đặt là tuỳ chọn chứ không bật mặc định, để mọi config cũ vẫn tái hiện đúng kết quả cũ.

## Hệ quả

1. **Mọi điểm offline trước ngày này đều bị thổi phồng** ở mức chưa biết, kể cả 0,911 /
   0,782 / 0,994. Không được trích các con số đó như năng lực thật của model.
2. **Các số đo trong game vẫn đúng** (tỉ số, số cú đập, đứng yên, bỏ lỡ bóng vào tầm): chúng
   đo hành vi, không đo model trên dữ liệu rò rỉ.
3. **Phép kiểm C++ khớp Python (0/21, 0/27) không phát hiện được lỗi này**: cả hai bên cùng
   đọc một trạng thái sai thời điểm, nên khớp nhau vẫn sai. Kiểm khớp chỉ chứng minh hai bên
   tính giống nhau, không chứng minh dữ liệu đúng thời điểm.
4. Các kết quả âm trước đó (feature vận tốc ADR-007, feature chuỗi ADR-010, DAgger vòng 1)
   được đo trên dữ liệu rò rỉ. Có thể cần đo lại.

## Bài học cho báo cáo

Khi dữ liệu huấn luyện được ghi từ một vòng lặp mô phỏng, phải kiểm **thời điểm ghi so với
thời điểm áp hành động**. Một cách kiểm rẻ: tại các dòng có nhãn hành động, nhân vật phải ở
trạng thái *có thể hành động*. Ở dự án này, phép kiểm đó cho 0% thay vì ~100%.

## Cập nhật 19/10/2026: `label_shift` bắt buộc

Mặc định cũ là `0`, nên chạy lại một config cũ là ra model rò rỉ mà không có cảnh báo. Từ nay
`train` và `eval_policy` **báo lỗi** nếu config schema v1/v2 không khai báo `label_shift`
(`train.check_label_shift`), và in cảnh báo nếu khai báo `0`. 14 config cũ được ghi rõ
`label_shift: 0` để vẫn tái lập đúng kết quả đã ghi trong `experiments.md`; test
`test_every_v1_v2_config_declares_label_shift` chặn config mới quên khai báo.

Các kết quả đo với `label_shift: 0` (model DAgger, kết quả âm của schema v2 ở ADR-010) chưa được đo
lại với `label_shift: 1`, nên chưa dùng làm kết luận.
