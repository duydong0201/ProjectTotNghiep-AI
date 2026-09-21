# ADR-001: Dùng Behavior Cloning thay vì Reinforcement Learning

- Trạng thái: Accepted
- Ngày: 2026-09-21

## Bối cảnh

Đề tài là xây dựng AI **học từ dữ liệu hành vi người chơi** cho game bóng chuyền 2D
*The Spike Cross Remaster*. Game đã có bot rule-based (`AIInputSystem`) và ghi log CSV mỗi trận.
Theo README của repo game, AI chỉ là một "Intent producer": nó xuất `CharacterIntent {moveX, finalIntent}`,
còn simulation tự kiểm tra tính hợp lệ.

Hiện trạng liên quan:
- Game **chưa có chế độ headless**. Mỗi trận phải chạy có đồ hoạ, theo thời gian thực.
- Người làm phần AI mới bắt đầu học ML, và thời gian có hạn (một học kỳ đồ án).

## Các lựa chọn đã xét

| Lựa chọn | Ưu | Nhược |
|---|---|---|
| **Behavior Cloning** (supervised: trạng thái → hành động của người) | Khớp đúng tên đề tài; chỉ cần log đã có; train vài giây trên laptop; model cây giải thích được | Chỉ giỏi tới mức người được bắt chước; lỗi tích luỹ (covariate shift) |
| Reinforcement Learning (PPO, DQN…) | Có thể chơi giỏi hơn người | Cần hàng triệu lượt chơi nên bắt buộc có headless + chạy song song; khó tinh chỉnh; không "học từ hành vi người" |
| Rule-based cải tiến | Đơn giản | Không phải ML, không đáp ứng đề tài |

## Quyết định

Dùng **Behavior Cloning** làm phương pháp chính. Bài toán được đặt thành hai bài phân loại:
`move ∈ {-1, 0, +1}` và `action ∈ FinalIntent`. Thứ tự model thử: majority baseline → Decision Tree
→ Random Forest → (LightGBM, MLP nếu cần).

Reinforcement Learning chỉ nằm ở mục "hướng phát triển" trong báo cáo.

## Hệ quả

- Chất lượng dữ liệu quyết định kết quả, nên phải chốt schema log v1 với bạn làm game (`docs/data_contract.md`).
- Phải xử lý mất cân bằng nhãn và lỗi tích luỹ. Các biện pháp: `class_weight`, action masking,
  đánh giá trong game sớm, và DAgger ở GĐ7.
- Việc đánh giá có hai tầng: offline (macro-F1 trên dev/holdout) và online (tỉ lệ thắng, độ giống người).
