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
| 2026-10-02 | spike_v0_human_20261002-142048 | v0 | player | spike_choice | majority | cv | 41 | 41 | 0.242 ± 0.205 | 0.131 ± 0.109 | 41 mẫu NGƯỜI chơi (Light 17 / Medium 19 / Strong 5). Mốc so sánh |
| 2026-10-02 | spike_v0_human_20261002-142048 | v0 | player | spike_choice | decision_tree | cv | 41 | 41 | 0.375 ± 0.250 | 0.289 ± 0.222 | 0.289 ± 0.222 - độ lệch chuẩn gần bằng chính điểm số, con số vô nghĩa |
| 2026-10-02 | spike_v0_human_20261002-142048 | v0 | player | spike_choice | random_forest | cv | 41 | 41 | 0.656 ± 0.100 | 0.485 ± 0.168 | Từng fold: [0.817, 0.429, 0.440, 0.385, 0.357] - fold cao gấp 2,3 lần fold thấp. So với 353 mẫu bot: 0.570 ± 0.043 (std nhỏ hơn 4 lần) |
| 2026-10-02 | spike_v0_human_20261002-142048 | v0 | player | spike_choice | logreg | cv | 41 | 41 | 0.556 ± 0.176 | 0.500 ± 0.164 | 0.500 ± 0.164. SpikeStrong chỉ 5 mẫu -> F1 của lớp này không có ý nghĩa |
| 2026-10-02 | spike_v0_human_20261002-160321 | v0 | player | spike_choice | majority | cv | 41 | 41 | 0.242 ± 0.205 | 0.131 ± 0.109 | Thêm 4 feature vận tốc (self_vx/vy, opp_vx, self_dx_10): 0.485 -> 0.483, KHÔNG đổi. Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_human_20261002-160321 | v0 | player | spike_choice | decision_tree | cv | 41 | 41 | 0.375 ± 0.250 | 0.289 ± 0.222 | Thêm 4 feature vận tốc (self_vx/vy, opp_vx, self_dx_10): 0.485 -> 0.483, KHÔNG đổi. Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_human_20261002-160321 | v0 | player | spike_choice | random_forest | cv | 41 | 41 | 0.656 ± 0.150 | 0.483 ± 0.172 | Thêm 4 feature vận tốc (self_vx/vy, opp_vx, self_dx_10): 0.485 -> 0.483, KHÔNG đổi. Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_human_20261002-160321 | v0 | player | spike_choice | logreg | cv | 41 | 41 | 0.467 ± 0.172 | 0.447 ± 0.117 | Thêm 4 feature vận tốc (self_vx/vy, opp_vx, self_dx_10): 0.485 -> 0.483, KHÔNG đổi. Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_bot_20261002-160429 | v0 | opponent | spike_choice | majority | cv | 353 | 353 | 0.549 ± 0.048 | 0.236 ± 0.014 | Thêm 4 feature vận tốc: 0.570 -> 0.568, KHÔNG đổi. self_vx và self_vy bằng 0 ở 100% frame đập bóng (bot dừng lại rồi mới đập). Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_bot_20261002-160429 | v0 | opponent | spike_choice | decision_tree | cv | 353 | 353 | 0.609 ± 0.050 | 0.533 ± 0.028 | Thêm 4 feature vận tốc: 0.570 -> 0.568, KHÔNG đổi. self_vx và self_vy bằng 0 ở 100% frame đập bóng (bot dừng lại rồi mới đập). Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_bot_20261002-160429 | v0 | opponent | spike_choice | random_forest | cv | 353 | 353 | 0.660 ± 0.075 | 0.568 ± 0.040 | Thêm 4 feature vận tốc: 0.570 -> 0.568, KHÔNG đổi. self_vx và self_vy bằng 0 ở 100% frame đập bóng (bot dừng lại rồi mới đập). Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_bot_20261002-160429 | v0 | opponent | spike_choice | logreg | cv | 353 | 353 | 0.593 ± 0.127 | 0.508 ± 0.064 | Thêm 4 feature vận tốc: 0.570 -> 0.568, KHÔNG đổi. self_vx và self_vy bằng 0 ở 100% frame đập bóng (bot dừng lại rồi mới đập). Đã hoàn lại, xem ADR-007 |
| 2026-10-02 | spike_v0_human_20261002-191756 | v0 | player | spike_choice | majority | cv | 41 | 41 | 0.267 ± 0.180 | 0.146 ± 0.094 | |
| 2026-10-02 | spike_v0_human_20261002-191756 | v0 | player | spike_choice | decision_tree | cv | 41 | 41 | 0.375 ± 0.250 | 0.217 ± 0.137 | |
| 2026-10-02 | spike_v0_human_20261002-191756 | v0 | player | spike_choice | random_forest | cv | 41 | 41 | 0.631 ± 0.088 | 0.493 ± 0.166 | |
| 2026-10-02 | spike_v0_human_20261002-191756 | v0 | player | spike_choice | logreg | cv | 41 | 41 | 0.556 ± 0.176 | 0.532 ± 0.178 | |
| 2026-10-05 | spike_v0_synthetic_20261005-230449 | v0 | player | spike_choice | majority | cv | 3896 | 3896 | 0.322 ± 0.008 | 0.162 ± 0.003 | DU LIEU TONG HOP (ADR-008), khong phai nguoi choi. 200 tran / 3.896 mau, 3 lop can nhau |
| 2026-10-05 | spike_v0_synthetic_20261005-230449 | v0 | player | spike_choice | decision_tree | cv | 3896 | 3896 | 0.916 ± 0.008 | 0.916 ± 0.008 | Du lieu tong hop. Tran ly thuyet ~0.92 (nhieu nhan 12%) |
| 2026-10-05 | spike_v0_synthetic_20261005-230449 | v0 | player | spike_choice | random_forest | cv | 3896 | 3896 | 0.917 ± 0.009 | 0.917 ± 0.009 | Du lieu tong hop: 0.917 = GAN CHAM TRAN 0.92 -> pipeline dung, moi that bai tren du lieu that KHONG phai do bug |
| 2026-10-05 | spike_v0_synthetic_20261005-230449 | v0 | player | spike_choice | logreg | cv | 3896 | 3896 | 0.807 ± 0.014 | 0.805 ± 0.013 | Du lieu tong hop. Thap hon cay vi quy luat chia theo dai, khong tuyen tinh |
| 2026-10-05 | spike_v0_human_20261005-235912 | v0 | player | spike_choice | majority | cv | 1401 | 1401 | 0.403 ± 0.011 | 0.192 ± 0.004 | |
| 2026-10-05 | spike_v0_human_20261005-235912 | v0 | player | spike_choice | decision_tree | cv | 1401 | 1401 | 0.534 ± 0.014 | 0.505 ± 0.021 | |
| 2026-10-05 | spike_v0_human_20261005-235912 | v0 | player | spike_choice | random_forest | cv | 1401 | 1401 | 0.544 ± 0.021 | 0.517 ± 0.028 | |
| 2026-10-05 | spike_v0_human_20261005-235912 | v0 | player | spike_choice | logreg | cv | 1401 | 1401 | 0.438 ± 0.005 | 0.390 ± 0.008 | |
| 2026-10-06 | spike_v1_20261006-015854 | v1 | player | spike_choice | majority | cv | 1466 | 1466 | 0.538 ± 0.025 | 0.233 ± 0.007 | DU LIEU TONG HOP (make_synthetic_v1.py), khong phai hanh vi nguoi choi. v1 lan dau, TRUOC khi sua nhan |
| 2026-10-06 | spike_v1_20261006-015854 | v1 | player | spike_choice | decision_tree | cv | 1466 | 1466 | 0.417 ± 0.016 | 0.389 ± 0.019 | Nhan lay tu target_pool (quota), doc lap voi moi feature -> khong hoc duoc |
| 2026-10-06 | spike_v1_20261006-015854 | v1 | player | spike_choice | random_forest | cv | 1466 | 1466 | 0.383 ± 0.016 | 0.376 ± 0.017 | 0.376 - thap hon ca v0 (0.517) du co 21 feature: them feature vo nghia khi nhan khong phu thuoc feature nao |
| 2026-10-06 | spike_v1_20261006-015854 | v1 | player | spike_choice | logreg | cv | 1466 | 1466 | 0.378 ± 0.016 | 0.372 ± 0.014 | Nhan tu quota |
| 2026-10-08 | spike_v1_20261008-185429 | v1 | player | spike_choice | majority | cv | 1483 | 1483 | 0.529 ± 0.022 | 0.230 ± 0.006 | DU LIEU TONG HOP (make_synthetic_v1.py), khong phai hanh vi nguoi choi. Sau khi sua CHI phan bot (sua phan player bi mat do script fail truoc write_text) |
| 2026-10-08 | spike_v1_20261008-185429 | v1 | player | spike_choice | decision_tree | cv | 1483 | 1483 | 0.430 ± 0.041 | 0.398 ± 0.024 | 0.389 -> 0.398: sua nua van chua du |
| 2026-10-08 | spike_v1_20261008-185429 | v1 | player | spike_choice | random_forest | cv | 1483 | 1483 | 0.374 ± 0.016 | 0.371 ± 0.017 | Chan doan: goi chinh spike_intensity() voi feature da ghi chi tai hien 26,7% nhan -> nhan khong den tu ham do |
| 2026-10-08 | spike_v1_20261008-185429 | v1 | player | spike_choice | logreg | cv | 1483 | 1483 | 0.362 ± 0.015 | 0.359 ± 0.015 | Van tu quota |
| 2026-10-08 | spike_v1_20261008-185658 | v1 | player | spike_choice | majority | cv | 1490 | 1490 | 0.471 ± 0.020 | 0.213 ± 0.006 | DU LIEU TONG HOP (make_synthetic_v1.py), khong phai hanh vi nguoi choi. Sau khi sua CA HAI ben: spike_intensity() tinh tai frame cham bong |
| 2026-10-08 | spike_v1_20261008-185658 | v1 | player | spike_choice | decision_tree | cv | 1490 | 1490 | 0.889 ± 0.019 | 0.876 ± 0.022 | 0.389 -> 0.876. Tran ly thuyet ~0.93 (nhieu nhan 10%) -> gan cham tran |
| 2026-10-08 | spike_v1_20261008-185658 | v1 | player | spike_choice | random_forest | cv | 1490 | 1490 | 0.868 ± 0.022 | 0.848 ± 0.024 | 0.848. Quota gio chi quyet dinh CO dap hay khong; dap CU NAO luon tinh tu trang thai |
| 2026-10-08 | spike_v1_20261008-185658 | v1 | player | spike_choice | logreg | cv | 1490 | 1490 | 0.825 ± 0.015 | 0.808 ± 0.015 | 0.808 - quy luat chia theo dai nen tuyen tinh cung bat duoc phan lon |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | move | majority | cv | 412107 | 412107 | 0.707 ± 0.008 | 0.276 ± 0.002 | DU LIEU TONG HOP (make_synthetic_v1.py), khong phai hanh vi nguoi choi. move + action tren 412.107 frame |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | move | decision_tree | cv | 412107 | 412107 | 0.841 ± 0.003 | 0.793 ± 0.004 | Xem random_forest cung run |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | move | random_forest | cv | 412107 | 412107 | 0.886 ± 0.004 | 0.842 ± 0.006 | move 0.842 (v0: 0.638) | action 0.926 (v0: 0.258 = bang majority). Cot bong mo ra ca hai. Tung lop action: None/Set/Serve 1.00, Jump 0.99, Bump 0.89, SpikeMedium 0.79 (yeu nhat vi la nhanh giua, nhan phan lon nhieu nhan) |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | move | logreg | cv | 412107 | 412107 | 0.615 ± 0.009 | 0.564 ± 0.009 | action 0.675 - tuyen tinh khong du cho 8 lop |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | action | majority | cv | 412107 | 412107 | 0.992 ± 0.000 | 0.125 ± 0.000 | DU LIEU TONG HOP (make_synthetic_v1.py), khong phai hanh vi nguoi choi. move + action tren 412.107 frame |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | action | decision_tree | cv | 412107 | 412107 | 0.999 ± 0.000 | 0.924 ± 0.007 | Xem random_forest cung run |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | action | random_forest | cv | 412107 | 412107 | 0.999 ± 0.000 | 0.926 ± 0.005 | move 0.842 (v0: 0.638) | action 0.926 (v0: 0.258 = bang majority). Cot bong mo ra ca hai. Tung lop action: None/Set/Serve 1.00, Jump 0.99, Bump 0.89, SpikeMedium 0.79 (yeu nhat vi la nhanh giua, nhan phan lon nhieu nhan) |
| 2026-10-08 | human_v1_20261008-191715 | v1 | player | action | logreg | cv | 412107 | 412107 | 0.992 ± 0.000 | 0.675 ± 0.007 | action 0.675 - tuyen tinh khong du cho 8 lop |
| 2026-10-08 | spike_v1_20261008-212417 | v1 | player | spike_choice | majority | cv | 1466 | 1466 | 0.538 ± 0.025 | 0.233 ± 0.007 | |
| 2026-10-08 | spike_v1_20261008-212417 | v1 | player | spike_choice | decision_tree | cv | 1466 | 1466 | 0.417 ± 0.016 | 0.389 ± 0.019 | |
| 2026-10-08 | spike_v1_20261008-212417 | v1 | player | spike_choice | random_forest | cv | 1466 | 1466 | 0.383 ± 0.016 | 0.376 ± 0.017 | |
| 2026-10-08 | spike_v1_20261008-212417 | v1 | player | spike_choice | logreg | cv | 1466 | 1466 | 0.378 ± 0.016 | 0.372 ± 0.014 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | move | majority | cv | 405456 | 405456 | 0.701 ± 0.010 | 0.275 ± 0.002 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | move | decision_tree | cv | 405456 | 405456 | 0.836 ± 0.005 | 0.789 ± 0.008 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | move | random_forest | cv | 405456 | 405456 | 0.879 ± 0.004 | 0.836 ± 0.007 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | move | logreg | cv | 405456 | 405456 | 0.604 ± 0.008 | 0.558 ± 0.011 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | action | majority | cv | 405456 | 405456 | 0.992 ± 0.000 | 0.125 ± 0.000 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | action | decision_tree | cv | 405456 | 405456 | 0.997 ± 0.000 | 0.730 ± 0.012 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | action | random_forest | cv | 405456 | 405456 | 0.998 ± 0.000 | 0.757 ± 0.008 | |
| 2026-10-08 | human_v1_20261008-212442 | v1 | player | action | logreg | cv | 405456 | 405456 | 0.992 ± 0.001 | 0.571 ± 0.020 | |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | move | majority | cv | 124172 | 124172 | 0.693 ± 0.007 | 0.273 ± 0.002 | 150 tran / 4 nguoi choi (P01-P04). cua so +/-20 frame quanh luc cham bong (giu 124.172/483.329 frame = 25,7%, giu 100% mau hanh dong) |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | move | decision_tree | cv | 124172 | 124172 | 0.835 ± 0.008 | 0.752 ± 0.012 | 0.752 |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | move | random_forest | cv | 124172 | 124172 | 0.840 ± 0.006 | 0.765 ± 0.011 | 0.765 - diem yeu con lai cua bo nay |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | move | logreg | cv | 124172 | 124172 | 0.676 ± 0.009 | 0.604 ± 0.013 | Tuyen tinh khong du |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | action | majority | cv | 124172 | 124172 | 0.971 ± 0.000 | 0.123 ± 0.000 | Doan toan None: acc 0.971 nhung macro-F1 0.123 |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | action | decision_tree | cv | 124172 | 124172 | 0.997 ± 0.000 | 0.892 ± 0.007 | 0.892 - chi kem RF 0.019 ma header nho hon ~100 lan |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | action | random_forest | cv | 124172 | 124172 | 0.998 ± 0.000 | 0.911 ± 0.014 | 0.911 = 7,4 lan baseline. Tung lop: None/Set/Serve/Jump 1.00, Bump 0.92, SpikeMedium 0.85, SpikeStrong 0.84, SpikeLight 0.69 |
| 2026-10-09 | human_v1_20261009-134123 | v1 | player | action | logreg | cv | 124172 | 124172 | 0.967 ± 0.003 | 0.602 ± 0.019 | 0.602 |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | move | majority | lopo | 124172 | 124172 | 0.694 ± 0.011 | 0.273 ± 0.003 | |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | move | decision_tree | lopo | 124172 | 124172 | 0.828 ± 0.008 | 0.743 ± 0.013 | |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | move | random_forest | lopo | 124172 | 124172 | 0.838 ± 0.008 | 0.761 ± 0.011 | LOPO 0.761 vs kfold 0.765 -> mat 0.004 |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | move | logreg | lopo | 124172 | 124172 | 0.675 ± 0.009 | 0.603 ± 0.012 | |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | action | majority | lopo | 124172 | 124172 | 0.971 ± 0.000 | 0.123 ± 0.000 | |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | action | decision_tree | lopo | 124172 | 124172 | 0.997 ± 0.000 | 0.899 ± 0.006 | |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | action | random_forest | lopo | 124172 | 124172 | 0.997 ± 0.000 | 0.908 ± 0.012 | LOPO 0.908 vs kfold 0.911 -> mat 0.003. Train 3 nguoi, test nguoi thu 4 chua tung thay -> model hoc quy luat chung, khong hoc thuoc tung nguoi. Luu y: 4 nguoi co the qua giong nhau nen con so nay co the de dai |
| 2026-10-09 | human_v1_lopo_20261009-134438 | v1 | player | action | logreg | lopo | 124172 | 124172 | 0.967 ± 0.002 | 0.602 ± 0.020 | |
