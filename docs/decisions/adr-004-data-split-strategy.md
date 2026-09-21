# ADR-004: Chiến lược chia dữ liệu và đánh giá theo từng loại dữ liệu

- Trạng thái: Accepted
- Ngày: 2026-09-21

## Bối cảnh

Ban đầu pipeline dùng một cách chung cho mọi dữ liệu: hash theo trận thành train / dev / holdout với tỉ lệ 60/20/20.
Kiểm tra phân bố thật trên 31 trận log v0 cho thấy cách này không đáng tin:

- **Độ dài trận chênh lệch rất lớn** (17 → 8.187 frame). Holdout chiếm 13% số trận nhưng chỉ 5,2% số frame.
  Riêng **một trận chiếm 78%** số frame của holdout.
- **Thiếu lớp:** holdout không có mẫu SpikeLight nào, SpikeStrong chỉ có 1 mẫu.
- **Một tập dev duy nhất cho số đo sai lệch:** trên 4 trận dev, RF `move` đạt macro-F1 0,668.
  Cross-validation 5 fold trên cả 31 trận cho **0,551 ± 0,040**.
- **Lỗi đo:** macro-F1 tính cả lớp có trong train nhưng vắng mặt ở tập đo, với F1 = 0.
  Ví dụ minh hoạ: 0,33 thay vì 0,495 dù model dự đoán y hệt.
- **Chia theo người chơi với ~8 người:** hash 20% có khoảng 17% khả năng holdout không có ai (0,8⁸).
  Nếu có thì thường chỉ 1–2 người, nên kết quả dao động rất lớn.
- Hai câu hỏi đánh giá khác nhau đang bị gộp làm một:
  (a) model bắt chước tốt những người đã có trong dữ liệu không;
  (b) model tổng quát sang người chơi mới không.

## Quyết định

**Giữ nguyên:** chia theo nhóm (trận hoặc người), không bao giờ chia theo frame. Hash tất định với salt cố định.
Holdout chỉ đo khi chạy `--final`.

**Chọn chế độ đánh giá (`split.dev_mode`) theo lượng dữ liệu:**

| Dữ liệu | Đánh giá hằng ngày | Holdout |
|---|---|---|
| v0 (31 trận, chỉ để chạy thông pipeline) | `kfold` 5 fold theo trận | không có (`holdout_ratio: 0`) |
| v1 bot (GĐ2, bot tự chơi được nhiều trận) | `single`: dev 20% theo hash | hash 20% theo trận |
| v1 người, câu hỏi (a) | `kfold` 5 fold theo trận | **holdout tương lai** (`holdout_dirs`) |
| v1 người, câu hỏi (b) | `lopo`: leave-one-player-out | holdout tương lai |

**Holdout tương lai:** các phiên chơi thu **sau khi đã chốt model**, lưu ở `data/raw/v1/human_holdout/`,
tốt nhất có vài người chơi chưa từng xuất hiện. Dữ liệu này chưa tồn tại lúc chọn model, nên về nguyên tắc
không thể bị tune vào. `train` từ chối chạy nếu holdout trùng `match_id` với dữ liệu train.

**Cách đo:**
- macro-F1 chỉ lấy trung bình trên lớp có mặt trong tập đo.
- Mỗi kết quả đi kèm support của từng lớp, và cảnh báo lớp có dưới 10 mẫu.
- CV báo cáo trung bình ± độ lệch chuẩn giữa các fold. Confusion matrix gộp từ dự đoán out-of-fold.

**Báo cáo cân bằng:** mỗi lần train ghi vào `split.json` số nhóm, số frame và phân bố nhãn của từng tập.
Cảnh báo khi một nhóm chiếm quá 50% một tập, hoặc khi tập đo thiếu lớp có trong train.

**Phiên bản game:** log v1 có cột `game_version`. Config lọc được bằng `game_versions: [...]`
để loại log từ bản game cũ.

## Các lựa chọn đã cân nhắc nhưng không chọn

- **Chia theo thứ hạng hash** (sắp xếp theo hash, lấy đúng N% đầu, như P-037): số lượng mỗi tập chính xác,
  nhưng thêm dữ liệu mới làm các nhóm cũ bị đổi tập. Holdout vì thế không giữ ổn định được.
- **Stratified group split** (phân tầng theo nhãn): khó kết hợp với hash ổn định. Chưa cần khi đã dùng CV.
  Có thể xem lại ở GĐ4 nếu dữ liệu người chơi quá lệch.
- **Holdout hash theo người chơi:** quá ít người, như đã phân tích ở trên.

## Hệ quả

- Số liệu trước ngày 21/09/2026 (dev chia ngẫu nhiên hoặc dev 4 trận) không so sánh trực tiếp được với số liệu CV.
  `reports/experiments.md` có ghi chú rõ ở các dòng này.
- CV tốn thời gian train gấp khoảng (n_folds + 1) lần. Với lượng dữ liệu hiện tại vẫn chỉ vài chục giây trên laptop.
- Phải lên lịch thu **holdout tương lai** ở GĐ6, sau khi chốt model (PLAN.md).
- Model export trong chế độ CV được fit trên toàn bộ dữ liệu không thuộc holdout.
