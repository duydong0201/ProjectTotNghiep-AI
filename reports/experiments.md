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
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | move | majority | dev | 57462 | 3175 | 0.680 | 0.270 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | move | decision_tree | dev | 57462 | 3175 | 0.730 | 0.645 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | move | random_forest | dev | 57462 | 3175 | 0.746 | 0.668 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | move | knn | dev | 57462 | 3175 | 0.743 | 0.616 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | action | majority | dev | 57462 | 3175 | 0.994 | 0.249 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | action | decision_tree | dev | 57462 | 3175 | 0.473 | 0.174 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | action | random_forest | dev | 57462 | 3175 | 0.685 | 0.218 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150821 | v0 | opponent | action | knn | dev | 57462 | 3175 | 0.994 | 0.249 | Dev = 4 trận, 1 trận chiếm 78%, thiếu SpikeLight -> KHÔNG đáng tin, xem run 150932 (CV) |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | move | majority | cv | 60637 | 60637 | 0.657 ± 0.029 | 0.264 ± 0.007 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | move | decision_tree | cv | 60637 | 60637 | 0.583 ± 0.049 | 0.529 ± 0.039 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | move | random_forest | cv | 60637 | 60637 | 0.605 ± 0.049 | 0.551 ± 0.040 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | move | knn | cv | 60637 | 60637 | 0.687 ± 0.018 | 0.537 ± 0.018 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | action | majority | cv | 60637 | 60637 | 0.995 ± 0.000 | 0.199 ± 0.000 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | action | decision_tree | cv | 60637 | 60637 | 0.583 ± 0.065 | 0.162 ± 0.012 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | action | random_forest | cv | 60637 | 60637 | 0.674 ± 0.037 | 0.175 ± 0.008 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-21 | baseline_v0_bot_20260921-150932 | v0 | opponent | action | knn | cv | 60637 | 60637 | 0.995 ± 0.000 | 0.199 ± 0.000 | CV 5 fold trên 31 trận (ADR-004); macro-F1 chỉ tính lớp có mặt |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | majority | dev | 183772 | 37785 | 0.770 | 0.290 | 101 trận, nhãn đã sửa (ADR-005). ĐO BẰNG dev đơn lẻ + macro-F1 kiểu cũ -> không so sánh được với các dòng cv |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | logreg | dev | 183772 | 37785 | 0.335 | 0.325 | Ranh giới quyết định không tuyến tính -> accuracy sụp, chỉ hơn majority chút ở macro-F1 |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | decision_tree | dev | 183772 | 37785 | 0.736 | 0.610 | Nhãn đã sửa (ADR-005). Cách đo cũ -> chỉ để đối chiếu nội bộ trong cùng run |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | random_forest | dev | 183772 | 37785 | 0.748 | 0.622 | Nhãn đã sửa (ADR-005). Cách đo cũ -> phải chạy lại bằng cv trước khi chọn model |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | move | knn | dev | 183772 | 37785 | 0.844 | 0.640 | Cao nhất nhưng KHÔNG export sang C++ được. Cách đo cũ |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | majority | dev | 183772 | 37785 | 0.996 | 0.200 | Đoán toàn None: acc 0.996 nhưng macro-F1 0.200 - minh hoạ vì sao không dùng accuracy |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | logreg | dev | 183772 | 37785 | 0.191 | 0.071 | Không học được |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | decision_tree | dev | 183772 | 37785 | 0.518 | 0.144 | Không học được |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | random_forest | dev | 183772 | 37785 | 0.738 | 0.180 | Không học được |
| 2026-09-25 | baseline_v0_bot_20260925-123104 | v0 | opponent | action | knn | dev | 183772 | 37785 | 0.996 | 0.200 | Bằng majority = không học được gì |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | move | majority | cv | 241542 | 241542 | 0.809 ± 0.084 | 0.297 ± 0.016 | CV 5 fold trên 101 trận, nhãn đã sửa (ADR-005). Mốc so sánh |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | move | logreg | cv | 241542 | 241542 | 0.557 ± 0.218 | 0.426 ± 0.056 | 0.325 -> 0.426 nhưng vẫn kém xa cây: ranh giới quyết định không tuyến tính |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | move | decision_tree | cv | 241542 | 241542 | 0.808 ± 0.105 | 0.624 ± 0.035 | So cùng cách đo (cv): 0.529 -> 0.624 nhờ sửa nhãn + 31->101 trận |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | move | random_forest | cv | 241542 | 241542 | 0.824 ± 0.094 | 0.638 ± 0.029 | 0.551 -> 0.638. Bằng knn trong sai số nhưng export C++ được -> chọn cho GĐ1 |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | move | knn | cv | 241542 | 241542 | 0.882 ± 0.062 | 0.639 ± 0.028 | 0.537 -> 0.639, cao nhất nhưng chênh RF chưa tới 1 sai số chuẩn; KHÔNG export C++ được |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | action | majority | cv | 241542 | 241542 | 0.997 ± 0.001 | 0.260 ± 0.120 | Đoán toàn None: acc 0.997, macro-F1 0.260 - vì sao không dùng accuracy |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | action | logreg | cv | 241542 | 241542 | 0.099 ± 0.063 | 0.049 ± 0.015 | Không học được |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | action | decision_tree | cv | 241542 | 241542 | 0.619 ± 0.192 | 0.224 ± 0.152 | Không học được; std 0.152 lớn hơn cả khoảng cách tới majority = nhiễu |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | action | random_forest | cv | 241542 | 241542 | 0.830 ± 0.085 | 0.258 ± 0.137 | Không học được: 0.258 ± 0.137, bằng majority. Phải chờ log v1 có bóng |
| 2026-10-01 | baseline_v0_bot_20261001-233311 | v0 | opponent | action | knn | cv | 241542 | 241542 | 0.997 ± 0.001 | 0.260 ± 0.120 | Bằng majority = không học được gì |
| 2026-10-02 | spike_v0_bot_20261002-002847 | v0 | opponent | spike_choice | majority | cv | 353 | 353 | 0.549 ± 0.048 | 0.236 ± 0.014 | Baseline. spike_choice chỉ dùng 353/241.542 frame có đập bóng (Light 128 / Medium 194 / Strong 31) |
| 2026-10-02 | spike_v0_bot_20261002-002847 | v0 | opponent | spike_choice | decision_tree | cv | 353 | 353 | 0.646 ± 0.087 | 0.565 ± 0.055 | Học được dù v0 thiếu bóng: bot luật đứng đúng điểm rơi nên self_x là proxy cho vị trí bóng |
| 2026-10-02 | spike_v0_bot_20261002-002847 | v0 | opponent | spike_choice | random_forest | cv | 353 | 353 | 0.655 ± 0.092 | 0.570 ± 0.043 | 0.236 -> 0.570. Tín hiệu đến từ self_x (importance 0.29) + 2 feature dẫn xuất từ nó. Xem ADR-006 |
| 2026-10-02 | spike_v0_bot_20261002-002847 | v0 | opponent | spike_choice | logreg | cv | 353 | 353 | 0.621 ± 0.106 | 0.547 ± 0.061 | Gần bằng cây -> quan hệ self_x với cú đập gần như đơn điệu |
