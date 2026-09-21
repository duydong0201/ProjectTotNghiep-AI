# Nhật ký thí nghiệm

Mỗi dòng do `spike_ai.train` tự thêm vào. Cột **Ghi chú** tự điền tay: đã thay đổi gì so với lần trước,
nhận xét về kết quả. Bảng này sẽ là nguồn chính cho chương "Thực nghiệm" của báo cáo.

| Ngày | Run | Schema | Agent | Target | Model | N train | N test | Accuracy | Macro-F1 | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | majority | 37615 | 23022 | 0.690 | 0.272 | |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | decision_tree | 37615 | 23022 | 0.545 | 0.499 | |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | random_forest | 37615 | 23022 | 0.564 | 0.516 | |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | knn | 37615 | 23022 | 0.659 | 0.506 | |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | majority | 37615 | 23022 | 0.994 | 0.199 | |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | decision_tree | 37615 | 23022 | 0.576 | 0.159 | |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | random_forest | 37615 | 23022 | 0.748 | 0.186 | |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | knn | 37615 | 23022 | 0.994 | 0.199 | |
