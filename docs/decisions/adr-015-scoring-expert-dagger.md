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

## Kết quả lần chạy 1 (20/10/2026, `selfplay_expert_v1`, 50 + 50 trận / vòng)

| Vòng | % điểm thắng | Cú đập trúng / trận | Set / trận | Jump / trận | Chuỗi Set→Jump→Spike | |
|---|---|---|---|---|---|---|
| xuất phát | 46,4% | 93,9 | 91,6 | 26,5 | 1.292 | |
| 1 | 47,9% | 123,2 | 125,1 | 38,7 | 1.889 | giữ |
| 2 | 49,9% | 126,2 | 127,1 | 38,7 | 1.917 | giữ |
| 3 | 49,9% | 123,3 | 125,0 | 37,8 | 1.861 | giữ |
| 4 | 48,7% | 132,7 | 134,0 | 40,7 | 2.020 | bỏ |
| 5 | **51,0%** | 123,3 | 125,5 | 38,4 | 1.856 | **giữ (tốt nhất)** |
| 6 | 50,7% | 124,4 | 126,0 | 39,5 | 1.949 | bỏ |
| 7 | 49,4% | 122,7 | 123,9 | 37,7 | 1.873 | bỏ |
| 8 | 50,6% | 131,8 | 133,3 | 40,4 | 1.986 | bỏ, tự dừng |

- **Lần đầu vòng tự động làm bot thắng nhiều hơn**: 46,4% -> 51,0% (+4,6 điểm %). Với ~1.150 điểm mỗi lần
  đo, chênh này cỡ 2 lần độ lệch chuẩn của hiệu số; cả 8 model mới đều nằm ở 47,9-51,0%, cao hơn mốc,
  nên tiến bộ là thật chứ không phải một lần may.
- Bot chơi giống chuyên gia hơn: cú đập trúng +31%, Set +37%, Jump +45%, chuỗi Set→Jump→Spike +44%.
- **Chững lại ở 50-51% từ vòng 2**, còn xa chuyên gia (79,2%).

**Nút thắt nghi ngờ:** trong game, bot ML không tự quyết định nhảy - việc nhảy sau Set do luật vật lý với
xác suất cố định 31,4% (ADR-012, `P_JUMP_AFTER_SET`). Chuyên gia thì lần nào cũng nhảy. Đo được: Jump chỉ
bằng ~30% số Set (38 / 125 mỗi trận). Model có học tốt mấy cũng không vượt được giới hạn đó.

Model tốt nhất: `models/selfplay_expert_v1_20261020-123720/best/` (train ở vòng 5). Chưa đưa vào game chính.

