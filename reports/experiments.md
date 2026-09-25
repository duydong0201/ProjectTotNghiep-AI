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
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | majority | dev | 183772 | 37785 | 0.770 | 0.290 | Sau ADR-004 (nhãn đã sửa) + import đủ 101 trận. Mốc so sánh |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | logreg | dev | 183772 | 37785 | 0.335 | 0.325 | Ranh giới quyết định không tuyến tính -> accuracy sụp, chỉ hơn majority chút ở macro-F1 |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | decision_tree | dev | 183772 | 37785 | 0.736 | 0.610 | 0.515 -> 0.610 so với nhãn hỏng. Export C++ được |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | random_forest | dev | 183772 | 37785 | 0.748 | 0.622 | 0.543 -> 0.622. Tốt nhất trong nhóm export được -> chọn cho GĐ1 |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | knn | dev | 183772 | 37785 | 0.844 | 0.640 | 0.554 -> 0.640, cao nhất nhưng KHÔNG export sang C++ được, chỉ dùng làm trần tham chiếu |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | majority | dev | 183772 | 37785 | 0.996 | 0.200 | Đoán toàn None: acc 0.996 nhưng macro-F1 0.200 - minh hoạ vì sao không dùng accuracy |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | logreg | dev | 183772 | 37785 | 0.191 | 0.071 | Không học được |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | decision_tree | dev | 183772 | 37785 | 0.518 | 0.144 | Không học được |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | random_forest | dev | 183772 | 37785 | 0.738 | 0.180 | Không học được |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | knn | dev | 183772 | 37785 | 0.996 | 0.200 | Bằng majority = không học được gì |
