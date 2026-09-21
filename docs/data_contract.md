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

## 3. Các cột (tóm tắt)

| Nhóm | Cột |
|---|---|
| Meta | `schema_version, match_id, frame, dt_ms` |
| Player (trái, entity 0) | `p_x, p_y, p_action_state, p_action_remain_ms, p_controller, p_player_id` |
| Opponent (phải, entity 3) | `o_x, o_y, o_action_state, o_action_remain_ms, o_controller, o_player_id` |
| Bóng | `ball_x, ball_y, ball_traj_type, ball_speed, ball_a, ball_b, ball_c, ball_landing_x, ball_state_frame, ball_collision_state` |
| Rally / trận | `rally_last_touch, rally_touch_count, score_left, score_right, serving_team` |
| Nhãn (input) | `p_input_move, p_input_intent, o_input_move, o_input_intent` |

Nguồn dữ liệu C++ của từng cột nằm ở trường `source` trong spec JSON.

## 4. Quy trình khi muốn đổi format

1. Sửa `schema/feature_spec.v1.json` (hoặc tạo `v2` nếu thay đổi lớn), tạo PR trên repo AI.
2. Hai bên review.
3. Bên game sửa `DatasetLogger`; bên AI sửa `features.py`. Test `tests/test_features.py`
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
