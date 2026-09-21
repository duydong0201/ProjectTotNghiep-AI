# Nhật ký thí nghiệm

Mỗi dòng do `spike_ai.train` tự thêm vào. Cột **Tập đo**: `dev` khi so sánh model hằng ngày, `holdout` chỉ xuất hiện khi chạy `--final` lúc đã chốt model. Cột **Ghi chú** tự điền tay: đã thay đổi gì so với lần trước,
nhận xét về kết quả. Bảng này sẽ là nguồn chính cho chương "Thực nghiệm" của báo cáo.

| Ngày | Run | Schema | Agent | Target | Model | Tập đo | N train | N đo | Accuracy | Macro-F1 | Ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | majority | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.690 | 0.272 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | decision_tree | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.545 | 0.499 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | random_forest | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.564 | 0.516 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | move | knn | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.659 | 0.506 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | majority | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.994 | 0.199 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | decision_tree | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.576 | 0.159 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | random_forest | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.748 | 0.186 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-134122 | v0 | opponent | action | knn | test (chia ngẫu nhiên cũ) | 37615 | 23022 | 0.994 | 0.199 | Trước khi chuyển sang chia theo hash |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | move | majority | dev | 44070 | 13392 | 0.690 | 0.272 | |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | move | decision_tree | dev | 44070 | 13392 | 0.581 | 0.515 | |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | move | random_forest | dev | 44070 | 13392 | 0.608 | 0.543 | |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | move | knn | dev | 44070 | 13392 | 0.726 | 0.554 | |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | action | majority | dev | 44070 | 13392 | 0.994 | 0.199 | |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | action | decision_tree | dev | 44070 | 13392 | 0.451 | 0.134 | |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | action | random_forest | dev | 44070 | 13392 | 0.610 | 0.161 | |
| 2026-09-21 | baseline_v0_bot_20260921-145400 | v0 | opponent | action | knn | dev | 44070 | 13392 | 0.994 | 0.199 | |
