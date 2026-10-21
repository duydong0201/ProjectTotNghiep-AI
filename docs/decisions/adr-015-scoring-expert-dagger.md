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

## Sửa chuyên gia (21/10/2026)

- **Vạch cuối:** game xét "ra ngoài" theo vị trí bóng ở FRAME chạm đất; frame cuối bóng còn đi thêm tới một bước
  (SpikeStrong ~28 px) sau điểm giao quỹ đạo với mặt đất. 145/170 điểm chuyên gia mất là SpikeStrong ra ngoài.
  Sửa: điểm rơi phải cách vạch cuối ít nhất một bước bay + cỡ bóng. Chuyên gia **83% -> ~97%**.
- **Thêm cách ghi điểm:** 25% cú Set xong đập từ mặt đất thay vì nhảy; rút thăm giữa các cú đập chắc ăn
  (thực tế gần như không tác dụng: sau khi có khoảng an toàn hầu như chỉ còn một cú chắc ăn).

## Train 2 giờ liên tục (21/10/2026): không vượt model v3

`selfplay_expert_v4` (3 vòng) và `selfplay_expert_v5` (chuyên gia luôn nhảy khi tự chơi, 21 vòng), đo trên
seed 800000 / xác nhận 900000: **24 vòng, không vòng nào được giữ**; model mới 63,6-69,9%, trung bình ~68%,
so với model v3 71,1% / 71,9%.

**Nguyên nhân (chẩn đoán model v3 và v5 vòng 3, 50 trận seed 800000):** ~85% số điểm bot ML mất là do **cú đập
của chính nó** (230/273 với v3, 248/283 với v5). Luật chặn của bot ML (`SpikeLandsInCourt`, "cú đập phải rơi
trong sân") dùng hàm đoán điểm rơi CŨ - đúng lỗi chuyên gia từng mắc (không tính bước bay cuối, không kiểm
tra qua lưới). Model học giỏi mấy, luật chặn vẫn để cú hỏng lọt qua.

## Sửa luật chặn của bot ML (21/10/2026) - không cần train lại

`SpikeLandsInCourt` dùng cùng phép kiểm tra đã sửa cho chuyên gia (repo game, `RunMode.safeSpikeMask`).
Cùng model v3:

| | luật chặn cũ | luật chặn mới |
|---|---|---|
| luôn nhảy, seed 800000 | 71,1% (mất 230 điểm do cú đập của mình) | **94,6%** (mất 0) |
| luôn nhảy, seed 400000 (để dành) | 69,4% (mất 265) | **95,8%** (mất 0) |
| nhảy 31,4%, seed 800000 | 51,4% (mất 469) | 53,4% (mất 351) |
| nhảy 31,4%, seed 400000 | 50,1% (mất 446) | 55,2% (mất 322) |

Với nhảy 31,4%, phần còn mất là cú đập từ mặt đất do **luật gốc** (`PickAttackIntent`) chọn - không sửa vì
luật gốc dùng chung với bot luật (đối thủ so sánh); sửa thì đối thủ cũng mạnh lên.

**Bài học cho báo cáo:** trước khi train thêm, phải xem điểm mất ở đâu. Hai giờ train không vượt được mốc vì
điểm nghẽn nằm ở luật chặn chứ không ở model; một sửa nhỏ ở luật chặn tăng +25 điểm %.

## Cú đập do luật gốc chọn cho bot ML (21/10/2026)

100 trận tới 15 điểm (model v3, luật chặn đã sửa, nhảy 31,4% từ dữ liệu người chơi): bot ML thắng 65/100
trận, nhưng 1.103/1.150 điểm bot luật ghi được là do bot ML tự đánh hỏng - **704 điểm là SpikeLight từ mặt
đất đập lưới**, do `PickAttackIntent` (luật gốc) chọn theo phép kiểm tra cũ. Sửa: chỉ phía bot ML dùng
phép kiểm tra đã sửa (`RunMode.safeRuleAttack`); bot luật giữ nguyên.

| 100 trận | trước | sau |
|---|---|---|
| seed 1000000: trận thắng / điểm | 65 / 53,2% | **100 / 74,0%** |
| seed 1100000: trận thắng / điểm | 61 / 52,8% | **99 / 75,4%** (1 trận bị cắt, bot luật 0 trận) |

Sau khi sửa: SpikeLight đập lưới 704 -> 2. Bot ML ghi điểm chủ yếu bằng **Set → Jump → Spike** (49%,
SpikeStrong 99%), bot luật đánh hỏng (39%), giao bóng ăn điểm (11%). Lỗi còn lại lớn nhất: **Set xong bóng
rơi lại sân mình** (407 điểm) - những lần không nhảy (68,6%) mà cũng không đập theo.

## Đập nhanh và v6 (22/10/2026): hai kết quả âm, giữ model v3

**Đập nhanh** (chuyên gia đập ngay quả đầu khi bóng vào vùng detect, `--expert-quick-spike`, repo game):
đo chính chuyên gia 100 trận (seed 600000 + 700000, ground 0.686):

| chuyên gia | % điểm |
|---|---|
| hiện tại (Set -> Nhảy -> Đập) | **96,7%** |
| đập nhanh 50% lượt | 82,1% |
| đập nhanh mọi lượt | 88,0% (chỉ ~1 điểm/trận: hai bên đập qua lại tới hết giờ) |
| đập nhanh chỉ khi đối thủ không kịp đón | 88,9% |

Đập từ điểm thấp ở quả đầu thì bot luật đỡ được; đập trên cao sau Set thì không. Không dùng (mặc định 0).

**`selfplay_expert_v6`** (chuyên gia chỉ nhảy ~31% sau Set như bot ML, còn lại đập từ đất; seed 1200000 /
1300000): model v3 73,5% / 73,7%; vòng 1-3: **45,6% / 51,2% / 56,4%**, cả 3 bị bỏ, dừng. Model mới đập trúng
ít hơn hẳn (114-142 cú/trận so với 186). Dạy "chờ bóng rồi đập từ đất" làm model kém đi; giữ v3.

## v7 (22/10/2026): chỉ học lượt chuyên gia ghi điểm + đập nhanh - ngang v3

`selfplay_expert_v7`: chuyên gia đập nhanh ở 50% lượt (chỉ khi dự đoán đối thủ không kịp đón), còn lại
luôn Set -> Jump -> Spike; train mới `expert_only_kills: true` chỉ giữ lượt chuyên gia tự ghi điểm bằng cú
đập. Vòng 1 được giữ (75,1% / 73,9% so với v3 74,1% / 71,9%), vòng 2-4 bỏ, dừng.

Đo thêm 50 trận mỗi seed, model v3 vs v7 vòng 1: seed 400000 (để dành) 72,6 vs 74,3%; 1000000 71,4 vs 73,2%;
1100000 74,2 vs 72,1%. **Trung bình 72,7 vs 73,2% - ngang nhau trong sai số.**

**Đập nhanh không ghi điểm được trước bot luật:** đếm cú đập ở quả chạm đầu tiên của đội:

| | đập quả đầu | ghi điểm | đập sau Set | ghi điểm |
|---|---|---|---|---|
| chuyên gia đập nhanh, seed 600000 | 486 | **0** | 7157 | 381 |
| bot ML v3, seed 400000 | 38 | 0 | 9655 | 237 |
| bot ML v7, seed 400000 | 221 | 1 | 8110 | 289 |

Bot ML v7 đập nhanh nhiều gấp ~6 lần nhưng gần như không ăn điểm; ngay cả chuyên gia (ước lượng đối thủ
không kịp) cũng 0/486 - bot luật chạy tới điểm rơi tức thì nên luôn đỡ được cú đập từ mặt đất ở quả đầu.
Mọi cú đập chỉ "chết" ~3% (đập sau Set); điểm chủ yếu đến từ đập trên cao và lỗi của bot luật.

## Chốt (22/10/2026): model v3 vào game chính

Repo game commit `d11ea0e` (nhánh `feat/scoring-expert`): `Source/AI/Generated/` = model
`selfplay_expert_v3_20261020-225201/best`, kèm luật chặn an toàn (`safeSpikeMask`, `safeRuleAttack`) và
nhảy 31,4% sau Set (đo từ dữ liệu người chơi, ADR-012). Bot luật vs bot ML: 73-75% điểm, thắng 99-100/100
trận. Game chính vẫn là người vs bot ML. **Chưa đo với người chơi thật** - việc tiếp theo của GĐ6.
