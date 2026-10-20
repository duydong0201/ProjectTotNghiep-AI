# ADR-015: Chuyên gia ghi điểm bằng quy tắc làm "người dạy" cho DAgger tự động

- **Ngày:** 20/10/2026
- **Trạng thái:** Accepted
- **Liên quan:** ADR-012 (luật nhảy), ADR-014 (self-imitation - kết quả âm)

## Bối cảnh

ADR-014: bắt chước chính mình (self-imitation) không làm bot thắng bot luật nhiều hơn; đối chứng xác
nhận lỗi không ở quy trình. Người dùng chọn hướng B: quy tắc ghi điểm viết tay làm chuyên gia, rồi
bot ML học theo, tự động hoàn toàn trong MatchRunner.

## Quyết định

**Chuyên gia** (`AIInputSystem::ExpertDecide`, repo game, nhánh `feat/scoring-expert`), đúng chuỗi người
dùng mô tả "đối thủ đánh sang -> Set -> Jump tới khi bóng vào vùng -> Spike tuỳ khoảng cách":

1. Trên không: bóng vào vùng đập thì đập cú tốt nhất.
2. Vừa Set xong: nhảy khi mô phỏng cho thấy sẽ chạm bóng sau 18-25 frame (cùng cửa sổ ADR-012),
   lần nào cũng nhảy.
3. Bóng vào vùng đập: đập ngay nếu đối thủ không kịp đón; còn chạm thì Set để nhảy đập. Không Bump.
4. **Cú đập tốt nhất** (`BestSpike`): trong các cú qua lưới và rơi trong sân, chọn cú có khoảng dư lớn
   nhất = |điểm rơi - đối thủ| - 7 px/frame × thời gian bay - 60 px tầm với.

Di chuyển vẫn theo luật như mọi bot. Chuyên gia chỉ dùng hành động và luật của game, cộng mô phỏng quỹ đạo.

**DAgger tự động:** khi tự chơi, mỗi lượt cầm bóng của bot ML được giao cho chuyên gia với xác suất
`expert_prob` (MatchRunner `--expert-prob`); log ghi `o_controller = expert (3)` (thêm vào enum
Controller của schema, không đổi feature). Train học frame `train_on_controllers: [human, expert]` với
`controller_weight`. Đo luôn tắt chuyên gia. `configs/selfplay_expert_v1.yaml`.

## Kiểm chứng chuyên gia trước khi cho bot học

50 trận đo (seed 100000), sân phải đấu bot luật:

| sân phải | tỉ số | % điểm |
|---|---|---|
| model hiện tại (bot ML) | 515 - 608 | 45,9% |
| chuyên gia, bản đầu | 742 - 410 | 64,4% |
| chuyên gia, thêm kiểm tra qua lưới | 750 - 184 | **79,2%** |

Bản đầu mất 248 điểm vì cú SpikeLight của chính nó đập lưới (`PredictAttackLandingX` bỏ qua lưới);
thêm kiểm tra độ cao bóng ở lưới thì hết. Còn mất 162 điểm do SpikeStrong (nghi ra ngoài sát vạch).

## Giới hạn (ghi trong báo cáo)

- Bot học theo **quy tắc do người viết code đặt ra**, không phải theo người chơi. Dữ liệu người chơi vẫn
  nằm trong tập train, nhưng phần tự động là học từ chuyên gia.
- Chuyên gia được tối ưu để thắng **bot luật**; chưa kiểm chứng với người chơi.
