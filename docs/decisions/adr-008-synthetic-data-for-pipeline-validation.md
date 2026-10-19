# ADR-008: Dữ liệu tổng hợp để kiểm chứng pipeline và dựng learning curve

- Trạng thái: Accepted
- Ngày: 2026-10-05

## Bối cảnh

Dữ liệu người chơi thật quá ít để kết luận: 90 mẫu `spike_choice` (Light 41 / Medium 40 /
Strong 9), đo ra macro-F1 0.485 ± 0.168 — độ lệch chuẩn gần bằng một phần ba điểm số.

Hai câu hỏi không trả lời được bằng chính dữ liệu đó:

1. **Pipeline có đúng không?** Nếu kết quả thấp, là do thiếu dữ liệu hay do có bug ở feature /
   nhãn / cách chia? Không phân biệt được.
2. **Cần bao nhiêu mẫu thì đủ?** Mục tiêu "150 mẫu mỗi lớp" trước đây là phỏng đoán, không có
   cơ sở đo đạc.

## Các lựa chọn đã xét

| Lựa chọn | Nhận xét |
|---|---|
| Chờ đủ dữ liệu người rồi mới biết | Không trả lời được câu hỏi 1, và không biết phải chờ đến bao nhiêu |
| Sinh dữ liệu giả ghi thẳng vào `Logs/` | **Từ chối.** File giả nằm lẫn file thật, cùng định dạng cùng kiểu tên -> sau một thời gian không ai phân biệt được dữ liệu nào là người chơi. Phá huỷ nguồn gốc dữ liệu, không thể khôi phục |
| Bot-vs-bot tự sinh | Đã thử, thất bại: rally lặp tuần hoàn, 78 cú đập nhưng chỉ 4 tình huống khác nhau |
| **Dữ liệu tổng hợp theo quy luật đã biết, thư mục riêng** | Trả lời được cả hai câu hỏi, và quy luật sinh nhãn là code commit trong repo nên ai cũng kiểm tra được |

## Quyết định

Thêm `scripts/make_synthetic_v0.py` sinh log định dạng v0 theo một quy luật **tường minh**
(`spike_policy()`): chọn cú đập theo dải khoảng cách tới lưới, hiệu chỉnh theo vị trí đối thủ,
kèm 12% nhiễu nhãn. Cả hai biến đó nằm trong 6 feature của v0 nên pipeline *phải* khôi phục được.

Ba rào chắn để dữ liệu này không bị nhầm thành dữ liệu thật:

1. Ghi vào `data/raw/v0_synthetic/`, tên file có tiền tố `synth_`. Script **từ chối chạy** nếu
   thư mục đích là `v0`, `v1` hay bất kỳ đường dẫn chứa `Logs`.
2. Quy luật sinh nhãn là code trong repo, không phải tham số ẩn. Có `seed` để tái lập.
3. Docstring đầu file và header config ghi rõ dùng được cho việc gì, không dùng được cho việc gì.

## Kết quả

**Sanity check** (200 trận, 643.358 frame, 3.896 mẫu, ba lớp cân nhau 1319/1252/1325):

| Model | macro-F1 |
|---|---|
| majority | 0.162 ± 0.003 |
| decision_tree | 0.916 ± 0.008 |
| **random_forest** | **0.917 ± 0.009** |
| logreg | 0.805 ± 0.013 |

Trần lý thuyết với 12% nhiễu nhãn là `1 − 0.12 + 0.12/3 ≈ 0.92`. Random Forest đạt **0.917**,
tức **gần như chạm trần**. Kết luận: feature, nhãn, cách chia, cách đo và export đều đúng.
Mọi thất bại trên dữ liệu thật **không phải do bug pipeline**.

**Learning curve** (`scripts/learning_curve.py`):

| N mẫu | Số trận | macro-F1 | std |
|---|---|---|---|
| 106 | 5 | 0.828 | **± 0.103** |
| 153 | 7 | 0.839 | ± 0.068 |
| 302 | 15 | 0.915 | ± 0.031 |
| 463 | 24 | 0.914 | ± 0.018 |
| 911 | 47 | 0.920 | ± 0.010 |
| 3896 | 200 | 0.917 | ± 0.009 |

Điểm trung bình **bão hoà từ khoảng N = 300**, nhưng **độ lệch chuẩn vẫn tiếp tục giảm**. Đây
là điểm quan trọng nhất: ở N = 106 thì std ± 0.103, con số đo được vô nghĩa dù trung bình đã
là 0.828.

## Hệ quả

- Mục tiêu dữ liệu không còn là phỏng đoán: **≈300 mẫu (100 mỗi lớp) để con số đáng tin,
  ≈450 để chắc chắn**. Hiện có 90 mẫu, tức khoảng **15 trận chơi tập trung** nữa.
- Con số 0.917 **không được trích dẫn** như kết quả học từ người chơi. Trong báo cáo nó thuộc
  mục "kiểm chứng phương pháp", kèm chú thích dữ liệu tổng hợp.
- Model train trên dữ liệu này **không đưa vào game** — nó chơi theo quy luật bịa, không theo người.
- Đường cong là của bài toán tổng hợp nên dễ hơn bài toán thật (v0 còn thiếu thông tin bóng,
  xem [ADR-007](adr-007-no-temporal-features-for-spike-choice.md)). Dùng để đọc **xu hướng và
  độ ổn định**, không dùng để dự đoán điểm tuyệt đối trên dữ liệu người.
