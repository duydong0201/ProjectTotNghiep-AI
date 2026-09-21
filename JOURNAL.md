# Journal — nhật ký tuần

> Mỗi tuần một mục: mục tiêu, đã làm được gì, khó khăn, bài học, kế hoạch tuần sau.
> Dùng để chuẩn bị buổi gặp giảng viên hướng dẫn, và làm tư liệu cho phần "Quá trình thực hiện" của báo cáo.
> Chi tiết từng việc đã xong nằm ở [WORKLOG.md](WORKLOG.md).

---

## Tuần 1: 21/09/2026 – 27/09/2026 (GĐ0 + bắt đầu GĐ1)

### Mục tiêu tuần này
- [x] Hiểu kiến trúc repo game (Input → Intent → Simulation) và định dạng log hiện có
- [x] Chọn hướng AI và lập kế hoạch cả kỳ
- [x] Dựng project AI và chạy được vòng đầu-cuối trên log v0
- [ ] Chốt schema log v1 với bạn làm game
- [ ] Tạo repo GitHub, push `main` + `develop`

### Đã hoàn thành
- Chọn Behavior Cloning (ADR-001), lập PLAN.md với 8 giai đoạn.
- Pipeline Python: đọc log → feature (có lật sân, ADR-003) → chia theo hash → train → export header C++ (ADR-002).
- Chạy trên 31 trận log v0: model `move` học được một phần (macro-F1 0.54 so với baseline 0.27).
- Quy trình làm việc: check script, CI, plan, ADR, worklog.

### Khó khăn & Giải pháp
| Khó khăn | Giải pháp | Kết quả |
|---|---|---|
| Log v0 không có thông tin bóng | Đề xuất schema v1 (`docs/data_contract.md`) | Chờ bạn làm game xác nhận |
| Nhãn `action` mất cân bằng nặng (None 99.45%) | Dùng macro-F1 thay accuracy, bật `class_weight` | Sẽ xử lý tiếp ở GĐ4 |
| m2cgen sinh code C mà MSVC không dịch được | Tự viết exporter C++ | ADR-002 |
| Console Windows không in được tiếng Việt | Ép stdout UTF-8, `PYTHONUTF8=1` trong check script | Đã sửa |

### Bài học
- Accuracy dễ đánh lừa khi dữ liệu mất cân bằng: model đoán toàn None vẫn đạt 99.4%.
- Phải chia dữ liệu theo trận chứ không theo frame, nếu không điểm sẽ cao ảo.
- Làm vòng đầu-cuối sớm giúp lộ ra vấn đề dữ liệu ngay từ tuần đầu.

### Kế hoạch tuần sau
- [ ] Họp với bạn làm game: chốt schema v1, fixed timestep, hook `MLInputSystem`
- [ ] Đưa header `move` vào game, `RunGoldenTest()` pass
- [ ] Học: Decision Tree, macro-F1, data leakage (Kaggle Learn: Intro to ML)

---

<!-- Copy block trên cho tuần tiếp theo -->
