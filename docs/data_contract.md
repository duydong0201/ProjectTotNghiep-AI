# Data contract giữa repo game và repo AI

Tài liệu này dành cho **cả hai người**: bạn làm game (C++) và bạn làm AI (Python).
Nguồn sự thật dạng máy đọc được là [`schema/feature_spec.v1.json`](../schema/feature_spec.v1.json).
Khi hai file lệch nhau, file JSON thắng.

## 1. Vì sao cần format mới (v1)

Log hiện tại (v0) của game chỉ có:

```text
Frame,PlayerPosX,PlayerPosY,PlayerEvent,BotPosX,BotPosY,BotEvent
```

Thiếu các thông tin sau thì model không thể học được *vì sao* người chơi hành động:

| Thiếu | Hậu quả |
|---|---|
| Vị trí / quỹ đạo / điểm rơi của bóng | Model không biết bóng ở đâu → không học được lúc nào nên chạy, lúc nào nên đánh |
| `touchCount`, `lastTouch`, trạng thái bóng | Không biết đang tới lượt ai |
| Cooldown / action state | Không phân biệt được "không muốn đánh" với "không thể đánh" |
| Input thật của frame (thay vì event đã apply) | Nhãn bị lệch thời gian |
| `match_id`, `player_id`, `controller` | Không chia train/test theo trận, không phân biệt người với bot |
| `game_version` | Log v0 trải qua 5 ngày (06–12/08) trong lúc code game/bot đang đổi, không lọc được log từ bản cũ |

Ngoài việc thiếu cột, v0 còn **ghi theo sự kiện chứ không theo frame**: mỗi nhân vật đang có
intent sinh ra một dòng riêng và nhân vật còn lại bị ghi cứng thành `"None"`, còn frame không ai
bấm phím thì không có dòng nào. Kết quả là 19,5% số dòng mang nhãn sai và nhãn "đứng yên" gần như
biến mất. `data.rebuild_frames_v0()` dựng lại một dòng mỗi frame trước khi tính feature — xem
[ADR-005](decisions/adr-005-rebuild-v0-frames.md). Quy tắc 1 của v1 dưới đây sinh ra chính là để
khỏi phải chữa cháy như vậy nữa.

## 2. Quy tắc ghi log v1 (phía game)

1. **Một dòng mỗi frame mô phỏng**, kể cả khi không có ai bấm gì.
2. **Thời điểm ghi:** sau khi mọi Intent của frame đã được sinh ra (`GenerateIntent::update`)
   và **trước** `ApplyIntentToComponent`. Như vậy trạng thái trong dòng log là trạng thái
   *lúc ra quyết định*, còn cột `*_input_*` là *quyết định* đó.
3. Nếu một nhân vật không có Intent trong frame thì `*_input_move = 0`, `*_input_intent = 0` (None).
4. Enum ghi dưới dạng **số nguyên** đúng với giá trị trong code C++ (bảng `enums` trong spec).
5. Mỗi trận là một file: `Logs/v1/<yyyy-mm-dd_hh-mm-ss>_<match_id>.csv`, header đúng thứ tự
   `raw_columns`.
6. **Fixed timestep:** dùng `SystemConfig::FIXED_DT` cho mô phỏng thay vì `delta` thật của
   Axmol (`MainScene::update` hiện dùng `FIXED_TIME = delta * 1000`). Nếu chưa đổi được,
   ghi `dt_ms` để phía AI biết.
7. Giữ nguyên log debug cũ. Dataset log là một output riêng (README game, mục 17).
8. `game_version` = git commit ngắn của repo game lúc build. Có thể nhúng lúc build bằng CMake:
   `execute_process(COMMAND git rev-parse --short HEAD ...)` rồi `add_compile_definitions(GAME_VERSION="...")`.
   Phía AI lọc theo phiên bản bằng `game_versions: [...]` trong config.
9. `match_id` phải **duy nhất trên mọi máy**, ví dụ `<yyyy-mm-dd_hh-mm-ss>_<player_id>`. Phía AI dùng nó để
   chia tập và để kiểm tra holdout không trùng trận với dữ liệu train.

### Trạng thái triển khai (06/10/2026)

`DatasetLogger` đã có bên game: `Source/System/DatasetLogger.{h,cpp}`, cấu hình ở
`Source/Config/Match/DatasetLogConfig.h`.

| Mục | Thực tế |
|---|---|
| Nơi ghi | `Logs/v1/<yyyy-mm-dd_hh-mm-ss>_<player_id>.csv`, thư mục con riêng để không lẫn log debug cũ |
| Thời điểm ghi | Sau `GenerateIntent::update`, trước `ApplyIntentToComponent` — đúng quy tắc 2 |
| Một dòng mỗi frame | Có. Kiểm chứng trên 6.780 frame: **0 frame trùng lặp, 0 lỗ hổng** |
| `game_version` | Commit ngắn của repo game, nhúng lúc build qua `target_compile_definitions` |
| `player_id` | `DatasetLogConfig::PLAYER_ID`, **phải đổi trước mỗi buổi thu với người mới** |
| Bật/tắt | `DatasetLogConfig::ENABLE` |
| `dt_ms` | ⚠️ Vẫn là `delta` thật (~17–19 ms), **chưa** dùng fixed timestep. Quy tắc 6 chưa làm |

Phía AI đã kiểm chứng: `detect_version` nhận ra `v1`, `validate_columns` pass, và
`features.build` tính ra đủ **21 feature** cho cả hai agent, không có NaN.

## 3. Các cột (tóm tắt)

| Nhóm | Cột |
|---|---|
| Meta | `schema_version, match_id, frame, dt_ms, game_version` |
| Player (trái, entity 0) | `p_x, p_y, p_action_state, p_action_remain_ms, p_controller, p_player_id` |
| Opponent (phải, entity 3) | `o_x, o_y, o_action_state, o_action_remain_ms, o_controller, o_player_id` |
| Bóng | `ball_x, ball_y, ball_traj_type, ball_speed, ball_a, ball_b, ball_c, ball_landing_x, ball_state_frame, ball_collision_state` |
| Rally / trận | `rally_last_touch, rally_touch_count, score_left, score_right, serving_team` |
| Nhãn (input) | `p_input_move, p_input_intent, o_input_move, o_input_intent` |

Từ 4 cột nhãn trên, phía AI sinh ra 3 target: `move`, `action`, và `spike_choice`.
`spike_choice` là **nhãn có điều kiện** — chỉ xác định ở frame `*_input_intent` là một trong ba cú
`SpikeLight / SpikeMedium / SpikeStrong`, các frame khác bị loại khỏi tập train và tập đo. Lý do và
giới hạn: [ADR-006](decisions/adr-006-conditional-spike-choice-target.md). Kiểm tra đã thu đủ mẫu
chưa bằng `python scripts/count_spikes.py --dirs data/raw/v1/human`.

### Bẫy nguồn gốc dữ liệu khi bật bot-vs-bot

Game có cờ `MatchRuleConfig::ENABLE_BOT_VS_BOT` cho hai bot tự đánh với nhau để sinh dữ liệu
hàng loạt. Khi bật, nhân vật sân trái **cũng là bot**, nhưng log v0 không có cột nào ghi lại điều
đó (v1 có `p_controller` / `o_controller`). Nếu trộn loại log này vào `data/raw/v0` rồi train với
`agent: player`, model sẽ học bot mà ta tưởng là học người.

Quy ước: log bot-vs-bot để riêng ở **`data/raw/v0_botvsbot/`**, không bao giờ nằm cùng thư mục với
dữ liệu người chơi. `scripts/count_spikes.py` gọi tên theo *phía sân* chứ không gọi là "người chơi",
để khỏi nhận lầm.

Hai việc **phía game** cần làm để `spike_choice` học được từ người chơi:
1. Ghi đủ cột bóng ở trên — không có chúng thì không có thông tin nào quyết định cú đập.
2. Bỏ `std::rand()` trong `AIInputSystem::PickAttackIntent`. Phần random nằm trong tập cú hợp lệ
   nên nó chặn cứng trần hiệu năng của mọi model (ADR-006).

Nguồn dữ liệu C++ của từng cột nằm ở trường `source` trong spec JSON.

## 4. Quy trình khi muốn đổi format

1. Sửa `schema/feature_spec.v1.json` (hoặc tạo `v2` nếu thay đổi lớn), tạo PR trên repo AI.
2. Hai bên review.
3. Bên game sửa `DatasetLogger`; bên AI sửa `features.py`. Test `tests/test_pipeline.py`
   bảo đảm thứ tự feature khớp spec.
4. Log cũ và mới không trộn chung một thư mục. Mỗi dòng có `schema_version` để phát hiện lỗi.

## 5. Phác thảo `DatasetLogger` (phía game)

```cpp
// Gọi trong MainScene::update, ngay sau _generateIntent->update(...)
// và TRƯỚC ApplyIntentToComponent(...)
void DatasetLogger::WriteFrame(int frame, float dtMs, ComponentStorage* s, IntentStorage* in)
{
    auto p   = s->GetCharacterPositionPool().get(GameConfig::PLAYER);
    auto o   = s->GetCharacterPositionPool().get(GameConfig::OPPONENT_1);
    auto pSt = s->GetCharacterActionStatePool().get(GameConfig::PLAYER);
    auto oSt = s->GetCharacterActionStatePool().get(GameConfig::OPPONENT_1);
    auto bp  = s->GetBallPositionPool().get(GameConfig::BALL);
    auto bt  = s->GetBallTrajectoryPool().get(GameConfig::BALL);
    auto bg  = s->GetBallGameplayStatePool().get(DEFAULT_MATCH);
    auto r   = s->GetRallyStatePool().get(DEFAULT_MATCH);
    auto m   = s->GetMatchGamePlayStatePool().get(DEFAULT_MATCH);
    // Lấy intent của frame (nếu có) cho PLAYER và OPPONENT_1 từ in->GetCharacterIntentPool()
    // rồi ghi đúng thứ tự cột trong raw_columns.
}
```
