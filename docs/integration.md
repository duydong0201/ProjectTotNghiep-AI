# Tích hợp model vào game C++ và kết nối hai repo

## 1. Hai repo nói chuyện với nhau qua 3 thứ

```text
  Repo game (C++)                               Repo AI (Python) - repo này
  ───────────────                               ─────────────────────────
  DatasetLogger ─── ① log CSV theo schema ───►  data/raw/  → spike_ai.train
                                                                   │
  MLInputSystem ◄── ② header .h đã export ───  exports/  ◄ spike_ai.export_cpp
        ▲
        └────────── ③ schema/feature_spec.*.json (thứ tự + công thức feature)
```

| Hướng | Cách chuyển | Ghi chú |
|---|---|---|
| Log: game → AI | `scripts/import_logs.py`, hoặc Google Drive nếu dữ liệu lớn | **Không** commit CSV vào git |
| Model: AI → game | copy `exports/spike_ai_*.h` vào `Source/AI/Generated/` của game | commit file .h vào repo game, ghi rõ version/run trong commit message |
| Model gốc (.joblib) | GitHub Releases của repo AI | tag theo run, vd `model-v1-move-dt-2026-10-15` |
| Schema | nằm trong repo AI; repo game tham chiếu tới | có thể dùng git submodule (mục 4) |

## 2. `MLInputSystem` phía game

Có cùng vai trò với `AIInputSystem`: đọc state, **chỉ sinh ra `CharacterIntent`**,
không sửa state trực tiếp (đúng nguyên tắc `Input -> Intent -> Simulation` của README game).

Ví dụ cho **GĐ1** (model `move` của schema v0, agent = opponent bên phải):

```cpp
#include "Generated/spike_ai_move_decision_tree.h"

namespace ml = spike_ai::move_decision_tree;

void MLInputSystem::update()
{
    auto st = _componentStorage->GetCharacterActionStatePool().get(GameConfig::OPPONENT_1);
    if (st->status != ActionState::None)      // action masking: đang cooldown thì không làm gì
        return;

    const float NET_X = BigRect::NET_X;       // = 1370, phải khớp constants.NET_X trong schema
    auto self = _componentStorage->GetCharacterPositionPool().get(GameConfig::OPPONENT_1)->position;
    auto opp  = _componentStorage->GetCharacterPositionPool().get(GameConfig::PLAYER)->position;

    // Opponent ở sân phải -> lật x qua lưới (y hệt features.py: x' = 2*NET_X - x)
    float selfX = 2.0f * NET_X - self.x;
    float oppX  = 2.0f * NET_X - opp.x;

    // ĐÚNG THỨ TỰ ml::kFeatureNames
    float x[ml::kNumFeatures] = {
        selfX,            // self_x
        self.y,           // self_y
        oppX,             // opp_x
        opp.y,            // opp_y
        oppX - selfX,     // dx_opp
        NET_X - selfX,    // self_dist_to_net
    };

    int move = std::atoi(ml::kClassNames[ml::PredictIndex(x)]);  // -1 / 0 / +1 (hệ đã lật)

    CharacterIntent intent{};
    intent.moveX = -move * SystemConfig::SPEED;  // lật ngược lại vì opponent ở bên phải
    if (intent.moveX != 0.0f)
        _intentStorage->GetCharacterIntentPool().add(GameConfig::OPPONENT_1, intent);
}
```

Ở GĐ1, model v0 chỉ điều khiển di chuyển. Giao bóng và đánh bóng có thể tạm giữ logic
của `AIInputSystem`. Từ GĐ2 (v1) thì model `action` quyết định cả `finalIntent`.

Bật/tắt: thêm một cờ cạnh `MatchRuleConfig::ENABLE_BOT`, ví dụ `BOT_MODE = Rule | ML`,
và chọn system tương ứng trong `GenerateIntent::update`.

### Golden test (bắt buộc mỗi lần cập nhật model)

Mỗi header có hàm `RunGoldenTest()`, dùng 20 vector lấy từ tập test mà Python đã tính sẵn kết quả.
Gọi một lần lúc khởi động (hoặc trong `Tests/`):

```cpp
AXASSERT(ml::RunGoldenTest(), "Model C++ lệch với Python - export lại!");
```

Golden test chỉ kiểm tra **model**. Để kiểm tra **cách tính feature trong C++**, ghi thêm vector
`x` ra log vài frame rồi so với `features.build()` trong Python trên cùng frame đó.

## 3. Checklist mỗi lần đưa model mới vào game

- [ ] `pytest` pass ở repo AI
- [ ] Export: `python -m spike_ai.export_cpp --model ... --out exports/`
- [ ] Copy file .h sang repo game, build thành công
- [ ] `RunGoldenTest()` trả về true
- [ ] Code tính feature trong `MLInputSystem` khớp `kFeatureNames` và `features.py`
- [ ] Chơi thử vài rally, ghi nhận xét vào `reports/experiments.md`

## 4. (Tuỳ chọn) Dùng git submodule

Nếu muốn repo game luôn trỏ đúng một commit của repo AI (để lấy schema và header):

```bash
# Chạy trong repo game
git submodule add https://github.com/<tai-khoan>/ProjectTotNghiep-AI.git External/AI
git commit -m "Add AI repo as submodule"

# Người khác clone repo game
git clone --recurse-submodules https://github.com/thanh551419a/ProjectTotNghiep.git

# Cập nhật lên commit mới nhất của repo AI
git submodule update --remote External/AI
```

Nếu dùng submodule, CMake của game có thể include thẳng `External/AI/exports/` thay vì copy tay.
Khi đó cần bỏ `exports/**` khỏi `.gitignore` của repo AI và commit các file .h đã chọn.
