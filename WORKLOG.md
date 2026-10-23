# Worklog — ProjectTotNghiep-AI

> Ghi lại **chức năng / outcome đã hoàn thành** kèm bằng chứng kiểm chứng được. Mỗi ngày một mục, mới nhất ở trên.
> Không ghi cho từng commit nhỏ. Nhật ký tuần (khó khăn, bài học) nằm ở [JOURNAL.md](JOURNAL.md).

---

## [2026-10-23]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Phân tích chuỗi Set -> Jump -> Spike | Done | Chỉ SpikeStrong trên không ghi điểm (chuyên gia 520/1105, bot ML 277/706; SpikeMedium 0/2298), và chỉ khi Set cách lưới ≥ 430 px | - |
| duydong0201 | Model tự quyết nhảy (vật lý chọn frame), repo game `--model-decides-jump` | Done | Model v3, seed 400000: 72,6% -> **93,8%**; seed 1000000: 71,4% -> 91,4%; thắng 95 -> 100/100 trận; chuỗi ghi điểm 524 -> 940. Không train lại | - |
| duydong0201 | `selfplay_expert_v8` (quy ước chuyên gia theo khoảng cách lưới) + `expert_kill_weight` | Kết quả âm | Mốc 92,0%; vòng 1-3: 84,9 / 71,4 / 56,1% | - |
| duydong0201 | `selfplay_expert_v9` (chuyên gia đập ngay 20%) | Kết quả âm | Model không học được đập ngay (16-20 cú/50 trận, ghi 0) | - |
| duydong0201 | `selfplay_expert_v10`: ép học đập ngay (`expert_quick_weight`, `keep_all`) | Done | Vòng 2: đập ngay ~60 lần/trận, 82,0% điểm (seed 400000). 4 test mới | - |
| duydong0201 | Đưa **v10 vòng 2** vào game chính, mặc định model tự quyết nhảy | Done | Repo game `2cc342c` (merge vào `feat/dong`). 100 trận: thắng **100/100**, **80,3%** điểm. 163 test pass | - |

**Tổng kết ngày:** Bước tăng lớn nhất lại không đến từ train thêm mà từ để model quyết việc nhảy (72,6% ->
93,8%). Đập ngay không ghi điểm được trước bot luật, nên chỉ học được khi ép bằng trọng số; chọn v10 vòng 2
vì lối chơi đa dạng hơn, đổi lại −12 điểm %. Việc tiếp theo: người thật chơi với bot ML, báo cáo.

## [2026-10-22]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Thử "đập nhanh" (chuyên gia đập ngay khi bóng vào vùng detect) | Kết quả âm | Chuyên gia 96,7% -> 82-89% điểm; cú đập quả đầu ghi 0/486 điểm (bot luật luôn đỡ được). Repo game: `--expert-quick-spike`, mặc định tắt. [ADR-015](docs/decisions/adr-015-scoring-expert-dagger.md) | - |
| duydong0201 | `selfplay_expert_v6`: chuyên gia nhảy ~31% như bot ML | Kết quả âm | v3 73,5% -> vòng 1-3: 45,6 / 51,2 / 56,4%, cả 3 bị bỏ | - |
| duydong0201 | `expert_only_kills` + `selfplay_expert_v7` | Done, ngang v3 | Chỉ học lượt chuyên gia tự ghi điểm bằng cú đập (2 test mới). 150 trận: v3 72,7% vs v7 73,2% (trong sai số) | - |
| duydong0201 | Đưa model v3 vào game chính | Done | Repo game commit `d11ea0e` (nhánh `feat/scoring-expert`, chưa push): v3 + luật chặn an toàn + nhảy 31,4%. Bot luật vs bot ML: 73-75% điểm, thắng 99-100/100 trận. Chưa đo với người | - |
| duydong0201 | Dọn repo: merge `develop` -> `main`, xoá 25 nhánh đã merge | Done | Còn `main` + `develop`; `check` PASS (ruff + 160 test) | - |

**Tổng kết ngày:** Trước bot luật chỉ có một cách ghi điểm hiệu quả: Set -> Jump -> Spike trên cao (đập
quả đầu luôn bị đỡ). Thử thêm kiểu ghi điểm không vượt v3, nên chốt v3 vào game chính. Việc tiếp theo:
cho người thật chơi với bot ML (≥ 20 trận) để có số liệu GĐ6, rồi viết báo cáo.

## [2026-10-21]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Xác nhận 2 bước khi chọn model (`confirm_seed`) và `selfplay_expert_v3` | Done | Model v3 tốt hơn trên cả 3 bộ trận độc lập (+3-4 điểm %); bộ để dành seed 400000: 69,4% (luôn nhảy) | - |
| duydong0201 | Sửa chuyên gia: cú SpikeStrong ra ngoài sát vạch cuối | Done | Điểm rơi cách vạch ≥ một bước bay + cỡ bóng: chuyên gia 83% -> ~97% | - |
| duydong0201 | Train liên tục 2 giờ (v4, v5) | Kết quả âm | 24 vòng, 0 vòng được giữ. Chẩn đoán: ~85% điểm bot ML mất do cú đập của chính nó | - |
| duydong0201 | Sửa luật chặn cú đập của bot ML (repo game) | Done | `safeSpikeMask`: v3 luôn nhảy 69,4% -> 95,8% (seed 400000). `safeRuleAttack`: SpikeLight đập lưới 704 -> 2; 100 trận tới 15 điểm: thắng 100/100 và 99/100, 74,0% / 75,4% điểm | - |
| duydong0201 | `selfplay.py --max-minutes / --stop-after`, chỉ lấy thư mục `iter_*` | Done | Có test tái hiện lỗi đọc nhầm `Logs/auto/xem_truc_tiep` | - |

**Tổng kết ngày:** Bài học lớn nhất: trước khi train thêm phải xem điểm mất ở đâu. Hai giờ train không
vượt mốc vì nút thắt nằm ở luật chặn chứ không ở model; sửa luật chặn tăng +25 điểm % mà không train lại.

## [2026-10-20]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Vòng tự học tự động bằng MatchRunner (bot vs bot không cửa sổ, ~1.000× thời gian thật) | Done | `spike_ai.selfplay`, `self_imitation` trong `train.py`, [ADR-014](docs/decisions/adr-014-self-imitation-selfplay.md) | - |
| duydong0201 | Self-imitation (học pha chính bot ML thắng / tự ghi điểm) | Kết quả âm | Không làm bot thắng nhiều hơn; đối chứng xác nhận lỗi không ở quy trình (ADR-014) | - |
| duydong0201 | Chuyên gia ghi điểm bằng quy tắc làm người dạy (DAgger tự động) | Done | `train_on_controllers`, `controller_weight`, `controller = expert (3)`. Seed mới: 46,4% -> 49,7% (nhảy 31,4%), 60,0% -> 69,3% (luôn nhảy). [ADR-015](docs/decisions/adr-015-scoring-expert-dagger.md) | - |
| duydong0201 | Phát hiện thiên lệch khi chọn model trên cùng bộ đo | Done | 51,0% trên seed dùng để chọn -> 49,7% trên seed mới; thêm `scripts/eval_fresh_seeds.py` | - |

## [2026-10-19]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Đo offline đúng cách bot ra quyết định trong game | Done | `policy.py`, `eval_policy.py`, 17 test mới, [ADR-013](docs/decisions/adr-013-evaluate-with-game-policy.md). 30 trận thật: macro-F1 argmax 0,110, bot trong game 0,141 | - |
| duydong0201 | Phát hiện dữ liệu sinh tự động trong `Logs/v1` | Cần quyết định | 298/338 file có 100% cú chạm ở `ball_y = 305` (vật lý không khớp game). Kết quả các ngày trước trên các tập đó cần đọc lại, xem ADR-013 | - |
| duydong0201 | Script đo bot trong game chính thức | Done | `python -m spike_ai.eval_ingame <log>`: % điểm thắng, cú đập bấm/trúng, đứng yên, bỏ lỡ bóng vào tầm, Set/Bump/Jump, chuỗi Set->Jump->Spike, Jump bừa, cách mất điểm. 11 test. Khớp số đo tay bot-vs-bot 16/10 (13-5, chuỗi 20, Jump bừa 0). Thay `scripts/analyze_botvsbot.py` | - |
| duydong0201 | Bắt buộc khai báo `label_shift` cho config v1/v2 | Done | Thiếu thì `train`/`eval_policy` báo lỗi, `0` thì cảnh báo rò rỉ nhãn; 14 config cũ ghi rõ `label_shift: 0` để tái lập; 5 test mới (ADR-011) | - |

## [2026-10-08]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | `DatasetLogger` ghi log v1 có cột bóng (phía game) | Done | `Source/System/DatasetLogger.{h,cpp}` + `Config/Match/DatasetLogConfig.h`. Gọi sau `GenerateIntent::update`, trước `ApplyIntentToComponent`. Kiểm chứng: 36 cột đúng schema, **0 frame trùng, 0 lỗ hổng**, phía AI tính được 21 feature không NaN | - |
| duydong0201 | Sửa 2 bug trong generator v1 | Done | Bot không bao giờ nhảy (`o_y` 1 → 1.723 giá trị); `p_peak_y` dùng trước khi gán | - |
| duydong0201 | Sửa nhãn `spike_choice` để học được | Done | `spike_intensity()` tính tại frame chạm bóng từ `self_dist_to_net`, `ball_y`, `dx_opp`. macro-F1 **0.389 → 0.876** | - |
| duydong0201 | Train đủ 3 target trên schema v1 | Done | `move` 0.842 ± 0.006, `action` **0.926 ± 0.005**, `spike_choice` 0.876 ± 0.022. Xem [experiments.md](reports/experiments.md) | - |

**Tổng kết ngày:** Log v1 có cột bóng đã hoạt động, và đây là bước mở ra toàn bộ phần còn lại.
So sánh cùng pipeline, cùng cách đo, chỉ khác schema:

| Target | v0 (306 trận, 6 feature) | v1 (150 trận, 21 feature) |
|---|---|---|
| `move` | 0.638 ± 0.029 | **0.842 ± 0.006** |
| `action` | 0.258 ± 0.137 (= majority) | **0.926 ± 0.005** |
| `spike_choice` | 0.517 ± 0.028 | **0.876 ± 0.022** |

`action` từ "bằng majority, hoàn toàn nhiễu" lên 0.926 — đúng như đã dự đoán ở [ADR-007](docs/decisions/adr-007-no-temporal-features-for-spike-choice.md):
hạn chế của v0 là **thiếu thông tin**, không phải thiếu mẫu, nên chỉ cột bóng mới giải quyết được.

**Hai bài học về phương pháp:**

1. **Nhãn phải phụ thuộc feature, và phải sinh ở đúng frame được ghi.** Lần đầu `spike_choice`
   trên v1 chỉ đạt 0.389 — thấp hơn cả v0 — vì nhãn lấy từ một danh sách quota độc lập với
   trạng thái, và nhánh dự phòng thì quyết định trước lúc chạm bóng 12+ frame bằng vị trí cũ,
   trong khi feature ghi vào log là vị trí sau khi đã chạy tới đón bóng.
2. **Cách chẩn đoán hiệu quả:** gọi chính hàm sinh nhãn với feature đã ghi rồi so với nhãn thật.
   Kết quả 26,7% (≈ đoán bừa 3 lớp) chỉ ra ngay nhãn không đến từ hàm đó. Hai lần đoán trước
   đều sai hướng.

**Lưu ý:** cả ba con số đo trên **dữ liệu tổng hợp theo quy luật đã biết**, thuộc mục "kiểm chứng
phương pháp". Giá trị thật: khi thu dữ liệu người qua `DatasetLogger`, nếu kết quả thấp thì chắc
chắn **không phải do pipeline hay feature** — trước đây không phân biệt được hai khả năng này.

**Việc tiếp theo:** thu dữ liệu người chơi thật qua `DatasetLogger` (đổi `DatasetLogConfig::PLAYER_ID`
cho mỗi người), và sửa action masking trong `AIInputSystem` để model không chọn cú đập bay ra ngoài biên.

## [2026-10-02]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Train `spike_choice` trên dữ liệu người chơi hiện có | Done | 41 mẫu (Light 17 / Medium 19 / Strong 5). RF macro-F1 **0.485 ± 0.168**, từng fold `[0.817, 0.429, 0.440, 0.385, 0.357]`. Xem [experiments.md](reports/experiments.md) run `spike_v0_human_20261002-142048` | - |
| duydong0201 | Đối chứng: cùng pipeline trên dữ liệu bot | Done | 353 mẫu → RF **0.570 ± 0.043**. Cùng feature, cùng model, cùng cách đo, chỉ khác số mẫu. Run `spike_v0_bot_20261002-002847` | - |
| duydong0201 | Chế độ bot-vs-bot phía game để tự sinh dữ liệu | Thử, không dùng | Hai bot tất định → rally lặp tuần hoàn: 78 cú đập nhưng chỉ 4 tình huống khác nhau. Ngẫu nhiên hoá quả giao + lệch vị trí đều không mở được bế tắc (5 thí nghiệm, bảng số liệu trong comment `AIInputSystem.cpp`). **Đã hoàn lại `std::rand()`** để game về đúng bản đã chốt | - |

**Tổng kết ngày — kết luận cho báo cáo:**

Đã train `spike_choice` (chọn 1 trong 3 cú `SpikeLight` / `SpikeMedium` / `SpikeStrong`) trên dữ liệu
người chơi hiện có. **Kết quả không dùng được, và lý do là thiếu dữ liệu**, có hai bằng chứng định lượng:

1. **Độ lệch chuẩn quá lớn so với điểm số.** RF đạt 0.485 nhưng ± 0.168, tức giá trị thật có thể nằm
   bất kỳ đâu trong khoảng 0.32–0.65. Decision tree còn tệ hơn: 0.289 ± 0.222 — độ lệch gần bằng
   chính điểm số. Từng fold của RF: `[0.817, 0.429, 0.440, 0.385, 0.357]`, fold cao gấp **2,3 lần**
   fold thấp. Một model học được quy luật thật thì không dao động như vậy.

2. **Đối chứng trực tiếp với dữ liệu nhiều hơn.** Chạy *đúng* pipeline đó trên 353 mẫu của bot:
   0.570 ± 0.043 — điểm cao hơn và **độ lệch chuẩn nhỏ hơn 4 lần**. Khác biệt duy nhất giữa hai
   run là số mẫu (41 so với 353). Đây là bằng chứng, không phải suy đoán.

Riêng lớp `SpikeStrong` chỉ có **5 mẫu** trên toàn bộ 101 trận, nên F1 của lớp này không có ý nghĩa
thống kê (`evaluate.py` tự cảnh báo khi lớp < 10 mẫu).

**Việc tiếp theo:** thu thêm dữ liệu cú đập của người chơi, mục tiêu ≥ 150 mẫu mỗi cú (hiện 17/19/5).
Không cần sửa game — chỉ cần chơi và chủ động đập bóng. Kiểm tra bằng `python scripts/count_spikes.py`.
Song song đó, hạn chế còn lại của log v0 là **thiếu cột bóng**: quyết định đập cú nào phụ thuộc vị trí
và quỹ đạo bóng, nên dù đủ mẫu vẫn cần log v1 ([data_contract.md](docs/data_contract.md)).

## [2026-10-01]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Gộp 3 nhánh đang treo vào `develop`, giải quyết trùng số ADR | Done | `develop` = merge của `fix/split-and-metrics`, `chore/pause-ci`, `fix/rebuild-v0-frames`; ADR trùng số 004 → đổi thành [ADR-005](docs/decisions/adr-005-rebuild-v0-frames.md); `check` PASS (ruff + 55 test) | - |
| duydong0201 | Import đủ log v0 và đo lại bằng cross-validation | Done | 101 trận / 241.542 frame. CV 5 fold, nhãn đã sửa: `move` RF **0.638 ± 0.029** (so cùng cách đo: 0.551 trước khi sửa nhãn, 31 trận). `action` vẫn 0.258 ± 0.137 = bằng majority. Xem [experiments.md](reports/experiments.md) run `20261001-233311` | - |

**Tổng kết ngày:** Đã có con số đáng tin đầu tiên cho `move`: Random Forest 0.638 ± 0.029 macro-F1, gấp
hơn hai lần baseline 0.297, và được đo bằng CV nên không phụ thuộc may rủi của một lần chia. KNN 0.639
nhúc nhích hơn nhưng chênh chưa tới một sai số chuẩn và không export sang C++ được, nên **chọn Random
Forest cho GĐ1**. `action` thì xác nhận dứt điểm là không học được ở v0 — độ lệch chuẩn (0.137) còn lớn
hơn khoảng cách tới majority, tức là hoàn toàn nhiễu. Việc tiếp theo: export RF ra header C++ và ghép
vào game để hoàn thành demo GĐ1.

## [2026-09-25]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Phát hiện log v0 ghi theo sự kiện chứ không theo frame, làm hỏng nhãn `move` | Done | Đo trên 209.552 dòng log: 40.838 dòng (19,5%) mang nhãn `None` giả; ~45% frame vắng mặt. Ví dụ tái hiện: `Logs/2026-08-06_15-46-15.csv` frame 416 | - |
| duydong0201 | Bước tiền xử lý dựng lại frame cho log v0 | Done | `data.rebuild_frames_v0()`, `tests/test_data.py` (10 test); [ADR-005](docs/decisions/adr-005-rebuild-v0-frames.md); `check` PASS (ruff + 32 test) | - |

**Tổng kết ngày:** Nguyên nhân model `move` yếu không chỉ là thiếu bóng như kết luận ngày 21/09,
mà còn vì nhãn bị gán sai có hệ thống. Mọi số đo trong `reports/experiments.md` trước ngày này
không còn dùng để so sánh được, cần train lại. Việc tiếp theo: import lại log và chạy lại loạt
so sánh model trên nhãn đã sửa.

## [2026-09-21]

| Member | Task | Status | Output / Bằng chứng | Time |
|---|---|---|---|---|
| duydong0201 | Phân tích repo game và đề xuất hướng Behavior Cloning | Done | [ADR-001](docs/decisions/adr-001-behavior-cloning.md), [PLAN.md](PLAN.md) | - |
| duydong0201 | Khởi tạo project AI: pipeline data → feature → train → export C++ | Done | `src/spike_ai/`; [docs/data_contract.md](docs/data_contract.md), [docs/integration.md](docs/integration.md) | - |
| duydong0201 | Vòng mỏng GĐ1 trên log v0 (31 trận, 60.637 frame) | Done (phía Python) | Model `move`: RF macro-F1 dev 0.543 vs baseline 0.272; `action` không học được do v0 thiếu bóng. Xem [experiments.md](reports/experiments.md). Chưa tích hợp vào game | - |
| duydong0201 | Exporter C++ riêng, có golden test | Done | [ADR-002](docs/decisions/adr-002-custom-cpp-exporter.md); `exports/spike_ai_move_decision_tree.h` (297 node). Chưa biên dịch thử bằng C++ | - |
| duydong0201 | Chia train/dev/holdout theo hash, có test chặn rò rỉ | Done | `src/spike_ai/split.py`, `tests/test_split.py`; `train --final` mới đo holdout | - |
| duydong0201 | Quy trình làm việc chắt lọc từ P-037: check script, CI, plan, ADR, worklog | Done | [docs/workflow.md](docs/workflow.md), `scripts/check.ps1`, `.github/workflows/ci.yml`; `check` PASS (ruff + 22 test) | - |
| duydong0201 | Sửa cách chia dữ liệu và cách đo macro-F1 theo phân bố thật của v0 | Done | [ADR-004](docs/decisions/adr-004-data-split-strategy.md); CV 5 fold: RF `move` macro-F1 **0.551 ± 0.040** (dev 4 trận cũ báo 0.668 — sai lệch); holdout tương lai `human_holdout/`; `check` PASS (45 test) | - |

**Tổng kết ngày:** Có đầy đủ đường ống Python chạy trên dữ liệu thật và được kiểm thử. Kết quả trên log v0
chứng minh vấn đề đã dự đoán: không có thông tin bóng thì không học được `action`. Kiểm tra phân bố thật cho thấy
một tập dev/holdout duy nhất trên 31 trận không đáng tin, nên chuyển sang cross-validation (ADR-004). Việc tiếp theo là chốt
schema v1 với bạn làm game, và thử header C++ trong game.
