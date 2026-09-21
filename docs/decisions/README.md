# Architecture Decision Records (ADR)

Mỗi quyết định kỹ thuật **đáng kể** là một file: bối cảnh, các lựa chọn, quyết định, hệ quả.
Các ADR này cũng là nguồn trực tiếp cho chương "Phương pháp" của báo cáo đồ án.

## Khi nào viết ADR?

Khi quyết định khó đảo ngược, ảnh hưởng nhiều phần, hoặc nhiều khả năng hội đồng sẽ hỏi "tại sao".
Không cần ADR cho chi tiết nhỏ như tên biến hay một hyperparameter.

## Quy ước

- Tên file: `adr-NNN-<slug-khong-dau>.md`, đánh số tăng dần, không dùng lại số.
- Trạng thái: `Proposed` → `Accepted` → (`Superseded by ADR-xxx` | `Deprecated`).
- **Không sửa nội dung ADR đã Accepted.** Đổi ý thì viết ADR mới rồi đánh dấu ADR cũ là Superseded.

## Mẫu

```markdown
# ADR-NNN: <Tiêu đề>

- Trạng thái: Proposed | Accepted | Superseded by ADR-xxx
- Ngày: YYYY-MM-DD

## Bối cảnh
## Các lựa chọn đã xét
## Quyết định
## Hệ quả
```

## Danh sách

| ADR | Tiêu đề | Trạng thái |
|---|---|---|
| [001](adr-001-behavior-cloning.md) | Dùng Behavior Cloning thay vì Reinforcement Learning | Accepted |
| [002](adr-002-custom-cpp-exporter.md) | Tự viết exporter C++ cho model cây thay vì dùng m2cgen | Accepted |
| [003](adr-003-agent-view-mirroring.md) | Chuẩn hoá góc nhìn bằng cách lật sân qua lưới | Accepted |
