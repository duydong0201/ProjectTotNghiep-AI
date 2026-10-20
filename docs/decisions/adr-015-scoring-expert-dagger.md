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

## Nút thắt là luật nhảy (20/10/2026)

Cùng model tốt nhất v1, chỉ đổi luật nhảy (MatchRunner `--jump-after-set`, `--model-may-jump`), 50 trận seed 100000:
nhảy sau Set 31,4% -> 51,0%; 60% -> 60,5%; **100% -> 71,7%**; để model tự quyết nhảy -> 35,9% (nhảy nhiều mà sai lúc).

Lần chạy v2 (`selfplay_expert_v2`, luôn nhảy cả lúc tự chơi lẫn lúc đo): 3 vòng 65,8 / 66,7 / 69,2% so với
mốc 71,7% -> bỏ cả 3, tự dừng.

## Đo lại trên bộ trận MỚI (seed 200000) - con số không thiên lệch

| Model | nhảy 31,4% | luôn nhảy |
|---|---|---|
| xuất phát (dagger_v1_w_aligned) | 46,4% | 60,0% |
| tốt nhất v1 (đang ở game chính) | 49,7% | 69,3% |
| v2 vòng 3 (bị bỏ, 69,2% trên seed cũ) | 49,2% | 72,3% |

- Tự học theo chuyên gia giúp thật: +3,3 điểm % (nhảy 31,4%), +9,3 điểm % (luôn nhảy) trên trận chưa từng dùng.
- 51,0% của model tốt nhất v1 bị thổi phồng (49,7% trên seed mới): chọn model cao nhất trên CÙNG một bộ đo
  thì con số được chọn lệch lên. Model v2 vòng 3 bị bỏ oan (72,3% so với 69,3%, chênh nằm trong độ nhiễu).
- Luật nhảy đóng góp nhiều nhất (+14-23 điểm %).

**Cần sửa quy trình:** chọn model trên một bộ seed, xác nhận trên bộ seed khác trước khi giữ
(`scripts/eval_fresh_seeds.py` đo lại trên seed mới).

## Lần chạy v3 (21/10/2026): xác nhận hai bước, luôn nhảy

`selfplay_expert_v3`: model mới phải thắng trên bộ đo chính (seed 300000) RỒI thắng tiếp trên bộ xác nhận
(seed 500000) mới được giữ. Bộ đo chính KHÔNG dùng seed 100000 nữa: model xuất phát đã được chọn trên
100000 nên con số ở đó bị thổi phồng (71,7% so với 67,0% trên 300000) - lần chạy đầu với 100000 bỏ gần
như mọi model mới, kể cả model thật ra tốt hơn.

| Vòng | bộ chính (300000) | bộ xác nhận (500000) | |
|---|---|---|---|
| xuất phát | 67,0% | 69,1% | |
| 1 | **71,2%** | **73,1%** | giữ |
| 2 | 67,6% | - | bỏ |
| 3 | 70,3% | - | bỏ |
| 4 | 70,2% | - | bỏ, tự dừng |

**Bộ trận để dành (seed 400000)**, không dùng để chọn gì (`scripts/eval_fresh_seeds.py`):

| Model | nhảy 31,4% | luôn nhảy |
|---|---|---|
| model đang ở game chính (tốt nhất v1) | 48,3% | 66,3% |
| v3 vòng 1 | **50,1%** | **69,4%** |

Model v3 tốt hơn trên cả 3 bộ trận độc lập (+4,2 / +4,0 / +3,1 điểm % khi luôn nhảy). Mức tăng thật cỡ
3-4 điểm %: nhỏ nhưng nhất quán. Model: `models/selfplay_expert_v3_20261020-225201/best/`.

