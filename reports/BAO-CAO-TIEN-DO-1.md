# KẾ HOẠCH VÀ TIẾN ĐỘ PHÁT TRIỂN AI BEHAVIOR CLONING

## 1. Mục tiêu

Mục tiêu của phần AI là xây dựng một mô hình **Behavior Cloning**, trong đó AI học cách thực hiện hành động dựa trên dữ liệu do người chơi tạo ra.

AI sẽ nhận thông tin trạng thái của game tại từng thời điểm và dự đoán hành động tương ứng của người chơi.

Lộ trình phát triển được chia thành các giai đoạn:

* Thu thập dữ liệu người chơi đầy đủ và chính xác.
* Kiểm chứng pipeline bằng dữ liệu từ bot có hành vi xác định.
* Huấn luyện AI từ dữ liệu người chơi.
* Kết hợp các model để AI có thể tự điều khiển nhân vật.
* Sau khi hệ thống ổn định, nghiên cứu DAgger để cải thiện khả năng bắt chước hành vi người chơi.

---

# 2. Tình trạng hiện tại

## 2.1. Pipeline Behavior Cloning

Đã xây dựng được pipeline cơ bản cho Behavior Cloning:

```text
Game
  ↓
Dataset Logger
  ↓
Preprocessing
  ↓
Training
  ↓
Random Forest
  ↓
Evaluation
  ↓
Export model sang C++
  ↓
AI sử dụng model trong game
```

Model hiện tại sử dụng **Random Forest** để dự đoán mức độ của hành động Spike:

* `SpikeLight`
* `SpikeMedium`
* `SpikeStrong`

Model sử dụng 6 feature chính liên quan đến vị trí của người chơi, đối thủ và khoảng cách đến lưới.

Kết quả hiện tại của model Spike:

* 101 trận
* 241.542 frame
* Chỉ có 41 mẫu Spike sau khi lọc dữ liệu
* 5-fold cross validation
* Macro-F1: **0,485 ± 0,168**

F1 theo từng lớp:

| Lớp        |   F1 |
| ----------- | ---: |
| SpikeLight  | 0,71 |
| SpikeMedium | 0,71 |
| SpikeStrong | 0,00 |

Nguyên nhân chính của kết quả chưa ổn định là số lượng dữ liệu Spike còn quá ít, đặc biệt `SpikeStrong` chỉ có 5 mẫu.

---

# 3. Vấn đề hiện tại

Vấn đề lớn nhất hiện tại nằm ở **dữ liệu đầu vào của Logger**.

Logger phiên bản hiện tại chưa ghi đầy đủ trạng thái của quả bóng và đang ghi dữ liệu theo sự kiện thay vì thực sự ghi một dòng cho mỗi frame.

Kết quả kiểm tra dữ liệu v0 cho thấy:

* Khoảng 19,5% dòng có vấn đề về nhãn.
* Khoảng 45% frame bị thiếu.
* Một số feature quan trọng về chuyển động của người chơi không thể thay thế cho thông tin về bóng.

Do đó, vấn đề hiện tại không chỉ nằm ở việc chọn model mà chủ yếu nằm ở **chất lượng và thông tin của dữ liệu**.

---

# 4. Việc cần làm trước mắt: Dataset Logger v1

Đây là bước quan trọng nhất trước khi mở rộng việc huấn luyện AI.

Logger v1 cần ghi dữ liệu **mỗi frame một dòng**, bao gồm trạng thái của game tại thời điểm AI đưa ra quyết định.

Các thông tin quan trọng cần bổ sung:

```text
ball_x
ball_y
ball_a
ball_b
ball_c
ball_landing_x
ball_state_frame

rally_last_touch
rally_touch_count

p_action_state
p_action_remain_ms

p_input_move
p_input_intent

match_id
player_id
controller
game_version
```

Trong đó, thông tin về quả bóng là phần quan trọng nhất vì AI cần biết vị trí và trạng thái của bóng để quyết định hành động.

### Hai nguyên tắc cần đảm bảo

**Thứ nhất: ghi mỗi frame một dòng.**

Không chỉ ghi khi người chơi bấm phím. Những frame không có hành động cũng cần được ghi lại để mô hình biết khi nào nên hành động và khi nào không.

**Thứ hai: ghi state trước khi áp dụng intent.**

Logger cần ghi trạng thái tại thời điểm AI/người chơi đưa ra quyết định, thay vì ghi sau khi hành động đã được áp dụng.

Điều này giúp tránh việc feature bị lệch thời điểm so với nhãn hành động.

---

# 5. Thu thập thêm dữ liệu người chơi

Song song với việc sửa Logger, tiếp tục thu thập dữ liệu người chơi.

Mục tiêu ban đầu:

| Intent      | Hiện tại | Mục tiêu |
| ----------- | ---------: | ---------: |
| SpikeStrong |          5 |        150 |
| SpikeLight  |         17 |        150 |
| SpikeMedium |         19 |        150 |
| Set         |         55 |        150 |
| Bump        |        774 |  Đã đủ |
| Jump        |          0 |        150 |

Đối với các hành động còn thiếu dữ liệu, người chơi sẽ chủ động sử dụng phím tương ứng trong các trận đấu để tạo thêm mẫu.

Mục tiêu dự kiến là khoảng **15–20 trận** để bổ sung dữ liệu.

Sau mỗi lần thu thập dữ liệu sẽ kiểm tra lại số lượng mẫu bằng script:

```text
count_spikes.py
```

Mục đích là đảm bảo dữ liệu thực sự được ghi nhận trước khi đưa vào training.

---

# 6. Lộ trình phát triển model

## Giai đoạn 1 — Kiểm chứng pipeline

Trước khi đánh giá dữ liệu người chơi, sử dụng dữ liệu từ **opponent bot** để kiểm tra pipeline.

Mục tiêu là để AI học lại hành vi của bot có luật cố định.

Nếu pipeline hoạt động đúng, model cần đạt kết quả cao, mục tiêu đặt ra là:

```text
Macro-F1 > 0.9
```

Nếu kết quả thấp, cần kiểm tra lại:

* Logger
* Feature
* Label
* Preprocessing
* Training pipeline

Chỉ sau khi bước này ổn định mới tiếp tục đánh giá dữ liệu người chơi.

---

## Giai đoạn 2 — Huấn luyện từ dữ liệu người chơi

Thay vì sử dụng một model duy nhất cho tất cả hành động, hệ thống dự kiến được chia thành hai tầng.

### Tầng A — Quyết định hành động

Model trả lời câu hỏi:

> "Ở frame này nhân vật có nên thực hiện hành động hay không, và nếu có thì thực hiện hành động nào?"

Các lớp bao gồm khoảng 9 loại action.

### Tầng B — Quyết định cường độ Spike

Nếu Tầng A quyết định thực hiện Spike, Tầng B tiếp tục trả lời:

> "Nên thực hiện Spike ở mức Light, Medium hay Strong?"

Tầng B chính là phần Spike model đã được xây dựng và kiểm thử.

Việc tách hai tầng giúp tránh tình trạng lớp `None` chiếm phần lớn dữ liệu khi tất cả frame được đưa vào cùng một model.

---

## Giai đoạn 3 — Kết hợp các model

Sau khi các model riêng lẻ hoạt động ổn định:

```text
State game
   ↓
Action Model
   ↓
Có hành động?
   ↓
 ┌───────────────┐
 │               │
Không            Có
 │               │
 ↓               ↓
Không làm     Chọn action
                   ↓
             Nếu là Spike
                   ↓
             Spike Model
                   ↓
        Light / Medium / Strong
                   ↓
              Game Agent
```

Kết hợp với model `Move`, mục tiêu là để AI có thể tự điều khiển nhân vật trong một trận đấu thay vì chỉ dự đoán từng hành động riêng lẻ.

---

# 7. Các hướng cải tiến sau khi pipeline ổn định

Sau khi có đủ dữ liệu và model cơ bản hoạt động, các cải tiến sẽ được thực hiện theo thứ tự:

### 7.1. Lấy mẫu quanh thời điểm chạm bóng

Thay vì sử dụng toàn bộ frame với tỷ lệ mất cân bằng rất lớn, tập trung dữ liệu quanh các thời điểm quan trọng như lúc chạm bóng.

Mục tiêu là giảm tình trạng một lớp chiếm quá nhiều dữ liệu.

### 7.2. Tối ưu Random Forest

Tiếp tục điều chỉnh các tham số như:

```text
class_weight = balanced
min_samples_leaf
```

Đây là bước có chi phí thấp vì vẫn sử dụng model hiện tại.

### 7.3. Thử LightGBM

Chỉ thực hiện khi Random Forest đạt giới hạn rõ ràng.

LightGBM có thể cho kết quả tốt hơn nhưng việc export model sang C++ sẽ phức tạp hơn.

### 7.4. Không ưu tiên MLP + ONNX ở giai đoạn hiện tại

MLP có thể tạo thêm dependency cho game trong khi lượng dữ liệu hiện tại còn nhỏ.

Do đó chưa ưu tiên hướng này.

### 7.5. DAgger

DAgger sẽ được nghiên cứu sau khi bot đã có khả năng tự chơi.

Behavior Cloning có một hạn chế là:

```text
Người chơi tạo dữ liệu
        ↓
AI học từ dữ liệu
        ↓
AI tự chơi
        ↓
AI đi vào tình huống chưa từng thấy
        ↓
AI mắc lỗi
        ↓
Tình huống tiếp theo càng khác dữ liệu ban đầu
```

DAgger giải quyết vấn đề này bằng cách cho AI tự chơi, sau đó người chơi sửa những tình huống AI xử lý sai và đưa các dữ liệu mới vào quá trình training.

---

# 8. Những vấn đề đã thử và kết quả

Một số hướng đã được thử nghiệm nhưng chưa mang lại kết quả:

| Phương pháp                                                     | Kết quả                                                                                                   |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Thêm feature vận tốc để bù cho việc thiếu thông tin bóng | Không cải thiện; các feature này bằng 0 ở 100% frame Spike                                           |
| Bot-vs-bot tự sinh dữ liệu                                      | Dữ liệu tạo ra các rally lặp lại, chỉ có khoảng 4 tình huống; nhiều cấu hình đều thất bại |
| Bỏ`rand()` để tạo nhãn học được                         | Nhãn bị suy biến về một lớp                                                                           |
| Train action nhiều lớp trên toàn bộ frame                     | Macro-F1 khoảng 0,258, tương đương baseline majority                                                  |

Các kết quả này cho thấy việc tiếp tục thêm feature hoặc đổi model khi **dữ liệu đầu vào chưa đầy đủ** chưa phải hướng giải quyết chính.

---

# 9. Cách đánh giá tiến độ

Để đảm bảo kết quả đánh giá có ý nghĩa, sẽ giữ các nguyên tắc:

### So sánh với baseline

Luôn so sánh model với `majority baseline`.

Ví dụ, một model có accuracy cao nhưng chỉ dự đoán lớp phổ biến nhất thì chưa chứng minh được rằng model đã học được hành vi.

### Sử dụng Macro-F1

Do dữ liệu mất cân bằng giữa các lớp, ưu tiên Macro-F1 thay vì chỉ sử dụng accuracy.

### Theo dõi độ ổn định

Không chỉ báo cáo giá trị trung bình mà báo cáo cả độ lệch chuẩn.

Ví dụ:

```text
0,485 ± 0,168
```

cho thấy kết quả giữa các fold còn dao động lớn.

### Giữ holdout độc lập

Holdout set chỉ được sử dụng để đánh giá cuối cùng sau khi đã quyết định model.

Không sử dụng kết quả holdout để liên tục điều chỉnh model.

---

# 10. Kết quả đã hoàn thành

Một số phần quan trọng của pipeline hiện đã hoàn thành:

* Đã xây dựng pipeline Behavior Cloning.
* Đã xây dựng model Random Forest cho Spike.
* Đã thực hiện Group-wise Cross Validation theo `match_id`.
* Đã kiểm tra prediction của model trên các trường hợp thực tế.
* Đã export model từ Python sang C++.
* Đã thực hiện Golden Test giữa Python và C++.
* Kết quả Python và C++ khớp trong sai số `1e-5`.

Như vậy, phần **pipeline huấn luyện → đánh giá → export → chạy trong C++** đã có nền tảng. Phần đang bị giới hạn hiện nay chủ yếu là **chất lượng và số lượng dữ liệu đầu vào**.

---

# 11. Trạng thái hiện tại và bước tiếp theo

Có thể tóm tắt tiến độ hiện tại như sau:

```text
[ĐÃ XONG]
Behavior Cloning pipeline
        ↓
[ĐÃ XONG]
Random Forest Spike
        ↓
[ĐÃ XONG]
Evaluation + Cross Validation
        ↓
[ĐÃ XONG]
Export Python → C++
        ↓
[ĐANG LÀM]
Logger v1 + bổ sung ball state
        ↓
[ĐANG LÀM]
Thu thập thêm dữ liệu người chơi
        ↓
[KẾ TIẾP]
Kiểm chứng pipeline bằng opponent bot
        ↓
[KẾ TIẾP]
Train Action Model
        ↓
[KẾ TIẾP]
Kết hợp Action + Move + Spike
        ↓
[MỤC TIÊU]
AI tự chơi được một trận
        ↓
[NẾU CÒN THỜI GIAN]
DAgger + model riêng từng người chơi
```

## Kết luận

Hiện tại dự án đã hoàn thành được phần nền tảng của pipeline AI và đã chứng minh được khả năng đưa model từ Python vào C++ để sử dụng trong game.

Tuy nhiên, kết quả hiện tại chưa đủ để kết luận AI đã bắt chước tốt hành vi người chơi. Nguyên nhân chính là dữ liệu Spike còn ít và Logger v0 chưa cung cấp đầy đủ thông tin trạng thái, đặc biệt là trạng thái của quả bóng.-

Vì vậy, ưu tiên hiện tại là **hoàn thiện Logger v1 và thu thập thêm dữ liệu**, sau đó dùng dữ liệu mới để kiểm chứng lại toàn bộ pipeline trước khi tiếp tục nâng cấp model.
