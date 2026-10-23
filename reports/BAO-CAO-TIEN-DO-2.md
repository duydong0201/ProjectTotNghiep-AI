# BÁO CÁO TIẾN ĐỘ 2: ĐÁNH GIÁ SO SÁNH BOT ML VÀ BOT RULE-BASED

---

## 📌 Tóm tắt tổng quan (Executive Summary)

Thực nghiệm so sánh giữa Bot Rule-based và Bot ML + Luật (Mô hình Behavior Cloning kết hợp Action Masking) trong cùng điều kiện thi đấu thực tế mang lại các kết luận cốt lõi sau:

1. **Hiệu quả thi đấu và tính chủ động vượt trội:**
   * Bot ML cải thiện tỷ lệ ghi điểm từ **78% lên 84%**.
   * Tần suất hành động trung bình tăng **gấp 6,4 lần** (từ 17,5 lên 112,4 hành động/trận).
   * Tần suất đập bóng tấn công (Spike) tăng mạnh **gấp 14,4 lần** (từ 5,8 lên 83,8 cú/trận).
2. **Hạn chế kỹ thuật cần hoàn thiện:**
   * Chiến thuật còn đơn điệu: 97,1% cú spike tập trung vào `SpikeMedium` (do cơ chế lấy nhãn xác suất cao nhất `argmax`).
   * Khả năng di chuyển chưa có sự khác biệt: Tỷ lệ đứng yên giữa 2 bot tương đương nhau (~71% – 72%).
   * Mô hình phụ thuộc vào lớp lọc: ML đề xuất thô có tỷ lệ vi phạm sân cao (~95%), cần lớp Action Masking để đảm bảo 100% cú đánh trong game hợp lệ.
3. **Đánh giá bản chất:** Mô hình ML đã học và tái hiện thành công phản xạ tấn công chủ động từ dữ liệu người chơi thật, giải quyết được trạng thái thụ động của bot luật; tuy nhiên hiện đang hoạt động hiệu quả dưới vai trò kết hợp trợ lực (hybrid) thay vì tự chủ chiến thuật độc lập.

---

## 1. Bảng so sánh chỉ số thực nghiệm (Đã chuẩn hóa)

> *Ghi chú phương pháp:* Nhằm đảm bảo tính khách quan do hai phiên đối chứng có độ dài khác nhau, toàn bộ chỉ số đếm thô đã được **chuẩn hóa theo Tỷ lệ phần trăm (%)** và **Tần suất trung bình trên mỗi trận đấu (TB/trận)**.

| Nhóm tiêu chí                   | Tiêu chí đánh giá                                 |  Bot Rule-based (Baseline)  |   Bot ML + Action Mask   | Mức độ cải thiện / Đánh giá                         |
| :--------------------------------- | :----------------------------------------------------- | :--------------------------: | :----------------------: | :---------------------------------------------------------- |
| **Hiệu quả thi đấu**     | **Tỷ lệ điểm thắng (% điểm ghi được)** |       **78,0%**       |     **84,0%**     | 🟢**+6,0%** (Tăng hiệu suất ghi điểm)            |
|                                    | Tỷ lệ để mất điểm                               |            22,0%            |          16,0%          | 🟢**Giảm 27,3%** tỷ lệ mất điểm                 |
|                                    | Số lần làm mất bóng TB                            |        3,3 lần/trận        |      2,6 lần/trận      | 🟢**Giảm 21,2%** sai sót phòng thủ                |
| **Mức độ chủ động**    | **Tần suất hành động TB**                   |   17,5 hành động/trận   | 112,4 hành động/trận | 🟢**Tăng gấp 6,4 lần** (Chủ động rõ rệt)      |
|                                    | **Tần suất đập bóng (Spike) TB**            |        5,8 cú/trận        |      83,8 cú/trận      | 🟢**Tăng gấp 14,4 lần** (Tấn công áp đảo)     |
|                                    | Tỷ trọng tấn công (Spike / Tổng action)           |            33,3%            |          74,6%          | 🟢 Tấn công trở thành lối chơi chủ đạo             |
|                                    | Tần suất giao bóng (Serve) TB                       |       11,7 lần/trận       |     28,0 lần/trận     | 🟢 Duy trì nhịp giao bóng ổn định                     |
| **Chất lượng kỹ thuật** | Tỷ lệ thời gian đứng yên                         |            71,9%            |          71,2%          | ⚪**Tương đương** (Chưa cải thiện di chuyển) |
|                                    | Tỷ lệ tập trung đòn đánh (`SpikeMedium`)      |  Phân bổ đều theo luật  |     **97,1%**     | 🔴**Đơn điệu** (Thiếu biến hóa chiến thuật)  |
|                                    | Kỹ thuật phối hợp (Set, Jump, Bump)                |        0,0 lần/trận        |      0,6 lần/trận      | 🔴 Chưa khai thác hiệu quả các kỹ thuật phụ         |
| **Tính hợp lệ hành vi**  | Tự chọn hành động hợp lệ (trước mask)         | 100% (cố định theo luật) |           ~5%           | 🔴 ML thô phụ thuộc vào luật chặn sân                |
|                                    | Tỷ lệ hành động hợp lệ khi thi đấu            |             100%             |           100%           | 🟢 Nhờ Action Masking lọc 100% lỗi vi phạm              |

---

## 2. Phân tích chi tiết ưu điểm của Bot ML

### 2.1. Nâng cao rõ rệt tính chủ động và áp lực tấn công

* Tần suất hành động tăng từ **17,5 lên 112,4 hành động/trận** (gấp 6,4 lần).
* Tần suất thực hiện các cú đập bóng (Spike) tăng từ **5,8 lên 83,8 cú/trận** (gấp 14,4 lần).
* **Ý nghĩa:** Khắc phục triệt để nhược điểm "thụ động, chỉ đón bóng khi rơi đúng tọa độ" của Bot Rule-based. Bot ML liên tục tạo áp lực và tìm cơ hội kết thúc pha bóng.

### 2.2. Giảm thiểu tỷ lệ mắc lỗi trực tiếp

* Tỷ lệ để đối thủ ghi điểm giảm từ **22,0% xuống 16,0%** (số lỗi mất bóng trung bình giảm từ 3,3 xuống 2,6 lần/trận).
* Mô hình phản xạ nhanh hơn trong các tình huống bóng nguy hiểm ở cự ly gần lưới.

### 2.3. Khả năng khái quát hóa từ dữ liệu mẫu

* Thay vì phải lập trình thủ công hàng trăm điều kiện `if-else` phức tạp cho từng tọa độ bóng, mô hình tự học không gian trạng thái từ hành vi của người chơi.
* Kiến trúc cho phép liên tục nâng cấp kỹ năng của bot chỉ bằng cách bổ sung dữ liệu thi đấu mới.

---

## 3. Các hạn chế kỹ thuật và nguyên nhân cốt lõi

| Điểm hạn chế                                            | Biểu hiện định lượng                                 | Nguyên nhân kỹ thuật                                                                                                                                                               |
| :---------------------------------------------------------- | :--------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Đòn đánh thiếu đa dạng**                     | 97,1% cú spike là`SpikeMedium` (407/419 cú)           | Cơ chế suy luận đang sử dụng`argmax` (luôn chọn nhãn có xác suất cao nhất), khiến bot chọn duy nhất hành động an toàn nhất mà không biến đổi đòn đánh. |
| **Phụ thuộc vào Action Masking**                   | ~95% đề xuất thô vi phạm biên sân phải             | Mô hình chưa học trọn vẹn ranh giới sân thi đấu từ tập dữ liệu; bắt buộc phải có lớp luật kiểm tra điều kiện vật lý trước khi thực thi.                   |
| **Mất cân bằng dữ liệu kỹ thuật phụ**         | Kỹ thuật`Bump`và Jump còn khá khiêm tốn ở bot ML | Tập dữ liệu huấn luyện có số lượng mẫu của các lớp này quá ít, khiến mô hình hầu như không thể dự đoán được.                                              |
| **Hiện tượng lệch phân phối (Covariate Shift)** | Từng gặp lỗi bot đứng bất động hoàn toàn         | Điểm đánh giá offline cao không phản ánh đúng thực tế in-game nếu quỹ đạo trạng thái trong game trôi dạt khỏi tập dữ liệu train ban đầu.                     |

---

## 4. Bảng đánh giá theo mục tiêu nghiên cứu

### 4.1. Mục tiêu đã đạt được hoàn toàn (Passed)

- [X] **Tích hợp pipeline end-to-end:** Model ML giao tiếp thời gian thực với engine game C++ không gây trễ khung hình.
- [X] **Cải thiện lối chơi:** Tăng tính chủ động tấn công (+14,4 lần tần suất Spike/trận) và nâng tỷ lệ thắng điểm lên **84%**.
- [X] **Đảm bảo an toàn hành vi:** Cơ chế Action Masking triệt tiêu hoàn toàn các hành động phạm luật của ML in-game.

### 4.2. Mục tiêu đạt một phần / Cần cải thiện (In Progress)

- [ ] **Tự chủ không gian (Spatial Awareness):** Cần cải thiện khả năng di chuyển đón điểm rơi bóng thay vì đứng chờ bóng vào tầm.
- [ ] **Đa dạng hóa chiến thuật (Tactical Diversity):** Chuyển từ cơ chế chọn cố định `argmax` sang lấy mẫu có trọng số (Weighted Probability Sampling) để bot phối hợp linh hoạt cả 3 mức Spike (`Light`, `Medium`, `Strong`).
- [ ] **Cân bằng nhãn dữ liệu:** Thu thập bổ sung dữ liệu cho các kỹ thuật hiếm (`Bump`, `Jump`, `Set`) hoặc áp dụng kỹ thuật sinh dữ liệu/tăng trọng số lớp (Class Weights).

---

## 5. Kết luận tổng quan (Dành cho báo cáo)

> ### 🎓 Kết luận báo cáo đồ án
>
> 1. **Về kết quả đạt được:** Nghiên cứu đã chứng minh tính khả thi của phương pháp **Behavior Cloning kết hợp Action Masking** trong việc điều khiển bot game thời gian thực. So với hệ thống thuần luật (Rule-based), bot ML thể hiện lối chơi vượt trội về tính chủ động tấn công, giảm 21,2% lỗi phòng thủ và đạt tỷ lệ điểm thắng **84,0%**.
> 2. **Về tính chất mô hình:** Bot hiện tại hoạt động hiệu quả theo mô hình **Lai ghép (Hybrid Architecture)**: ML đóng vai trò "bộ não tạo phản xạ tấn công" dựa trên dữ liệu người chơi, trong khi Action Masking đóng vai trò "hành lang an toàn" ngăn chặn các hành vi vi phạm luật vật lý.
> 3. **Hướng phát triển tiếp theo:** Để tiến tới mô hình tự chủ hoàn toàn, giai đoạn tiếp theo cần tập trung vào:
>    * Thay đổi cơ chế ra quyết định từ `argmax` sang phân phối xác suất ngẫu nhiên có kiểm soát để đa dạng hóa cú đánh.
>    * Triển khai thuật toán **DAgger (Dataset Aggregation)** bằng cách đưa chính dữ liệu bot tự thi đấu vào tập huấn luyện nhằm khắc phục triệt để hiện tượng lệch phân phối trạng thái (Covariate Shift).
