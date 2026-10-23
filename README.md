# ProjectTotNghiep-AI — Bot bóng chuyền học bằng Behavior Cloning

Phần AI/ML của đồ án tốt nghiệp **The Spike Cross Remaster**, game bóng chuyền 2D viết bằng C++/Axmol
(repo game: [thanh551419a/ProjectTotNghiep](https://github.com/thanh551419a/ProjectTotNghiep)).

Repo này huấn luyện một **policy** điều khiển nhân vật bot: nhìn trạng thái game ở mỗi frame và chọn hành động
(Set, Bump, Jump, các cú đập, giao bóng), bằng cách **bắt chước** dữ liệu có nhãn từ người chơi và từ một
chuyên gia viết bằng quy tắc. Model được xuất thành **một file header C++ thuần**, game gọi trực tiếp, không
cần thư viện ML.

```text
trạng thái game (21 feature)  ──►  Random Forest  ──►  hành động  ──►  luật chặn an toàn  ──►  game
```

## Kết quả hiện tại

Model đang dùng trong game chính: **`selfplay_expert_v10` vòng 2** (từ 23/10/2026). Đo bằng 100 trận bot
luật (sân trái) đấu bot ML (sân phải), chạy không cửa sổ:

| Chỉ số | Bot ML |
|---|---|
| Trận thắng | **100 / 100** |
| % điểm thắng | **80,3%** (1381 – 338) |
| Điểm mất mỗi trận | 3,4 |
| Lối chơi | Set → Jump → Spike, đập ngay khi bóng vào vùng detect (~62 lần/trận), Bump cứu bóng |

Chưa đánh giá với người chơi thật. Tiến triển qua từng mốc và các kết quả âm:
[reports/experiments.md](reports/experiments.md), [ADR-015](docs/decisions/adr-015-scoring-expert-dagger.md).

## Cách hoạt động

| Thành phần | Mô tả |
|---|---|
| **Dữ liệu** | Game ghi mỗi frame một dòng (schema v1: nhân vật, bóng, quỹ đạo, điểm số, phím bấm, loại cú chạm) vào `Logs/v1/*.csv` |
| **Feature** | 21 feature tính lại được y hệt trong C++ (chỉ cộng/trừ, so sánh, lật sân qua lưới) – [`schema/feature_spec.v1.json`](schema/feature_spec.v1.json) |
| **Nhãn** | Hành động của frame kế tiếp (`label_shift: 1`), vì game ghi log sau khi áp hành động – [ADR-011](docs/decisions/adr-011-label-leakage-from-logging-order.md) |
| **Model** | Random Forest (100 cây), xuất sang header C++ có golden test so khớp Python – [ADR-002](docs/decisions/adr-002-custom-cpp-exporter.md) |
| **Trong game** | Model chọn hành động và quyết có nhảy sau Set; vật lý chọn frame nhảy; luật chặn loại cú đập ra ngoài/đập lưới – [ADR-012](docs/decisions/adr-012-hybrid-model-decides-physics-times-jump.md) |
| **Vòng tự học (DAgger)** | `MatchRunner` (bot vs bot không cửa sổ, ~1.000× thời gian thật) tự chơi, một nửa số lượt do chuyên gia quy tắc điều khiển → train lại → export → build → đo trên bộ trận mới → chỉ giữ model tốt hơn – [ADR-014](docs/decisions/adr-014-self-imitation-selfplay.md), [ADR-015](docs/decisions/adr-015-scoring-expert-dagger.md) |
| **Đánh giá** | Chia dữ liệu theo trận/người chơi; đo offline đúng cách bot quyết định trong game ([ADR-013](docs/decisions/adr-013-evaluate-with-game-policy.md)); đo trong game trên bộ seed chưa dùng để chọn model |

## Cài đặt

Yêu cầu: Python 3.10+ (đang dùng 3.11), Windows. Vòng tự học cần thêm repo game và Visual Studio 2022 để build
`MatchRunner`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

`check.ps1` chạy ruff và toàn bộ test (163 test). Phải PASS trước mỗi commit.

## Cách dùng

```bash
# Train theo một config (dữ liệu đọc thẳng từ Logs/v1 của repo game), lưu vào models/, ghi experiments.md
python -m spike_ai.train --config configs/played_v1_aligned.yaml

# Đo offline theo đúng cách bot ra quyết định trong game (ngưỡng, rút thăm, luật chặn)
python -m spike_ai.eval_policy --config configs/played_v1_aligned.yaml

# Đo bot trong game từ log trận: % điểm, cú đập, Set/Bump/Jump, chuỗi Set->Jump->Spike, cách mất điểm
python -m spike_ai.eval_ingame ../ProjectTotNghiep/Logs/v1/<trận>.csv

# Vòng tự học với chuyên gia (cấu hình của model đang dùng); tối đa 90 phút
python -m spike_ai.selfplay --config configs/selfplay_expert_v10.yaml --max-minutes 90

# Đo lại một model trên bộ trận mới, không dùng để chọn model
python scripts/eval_fresh_seeds.py --seed 400000 --model "v10=models/<run>/iter02/spike_ai_action_random_forest.h"

# Xuất model sang header C++ (kèm golden test)
python -m spike_ai.export_cpp --model models/<run>/action__random_forest.joblib --out exports/
```

Đưa model mới vào game: làm theo checklist ở [docs/integration.md](docs/integration.md) (đo trên bộ trận mới,
commit riêng ở repo game để revert được).

## Cấu trúc thư mục

```text
ProjectTotNghiep-AI/
├── src/spike_ai/          # package chính
│   ├── schema.py, data.py #   đọc spec, đọc log CSV / CSV.gz, gắn match_id
│   ├── features.py        #   raw -> feature + nhãn (phải khớp C++)
│   ├── split.py           #   chia theo hash trận/người chơi, k-fold, LOPO
│   ├── models.py          #   majority, decision tree, random forest, ...
│   ├── train.py           #   CLI train: lọc người dạy, trọng số, ghép nhãn
│   ├── evaluate.py        #   macro-F1, confusion matrix
│   ├── policy.py          #   cách bot ra quyết định trong game (chép từ AIInputSystem)
│   ├── eval_policy.py     #   CLI đo offline theo policy đó
│   ├── eval_ingame.py     #   CLI đo bot trong game từ log trận
│   ├── selfplay.py        #   CLI vòng tự học (gọi MatchRunner bên repo game)
│   └── export_cpp.py      #   CLI xuất model -> header C++ + golden test
├── configs/               # YAML cho mỗi lần train / vòng tự học (tái lập được)
├── schema/                # nguồn sự thật: cột log, enum, thứ tự feature
├── scripts/               # check.ps1/.sh, công cụ dữ liệu, đo trên seed mới
├── tests/                 # pytest
├── docs/                  # workflow, hợp đồng dữ liệu, tích hợp C++, ADR
├── reports/experiments.md # nhật ký mọi thí nghiệm, kể cả kết quả âm
├── PLAN.md, WORKLOG.md, JOURNAL.md
├── data/, models/, exports/   # không commit
```

## Tài liệu

| Cần tìm | Ở đâu |
|---|---|
| Kế hoạch và tình trạng từng giai đoạn | [PLAN.md](PLAN.md) |
| Việc đã xong kèm bằng chứng | [WORKLOG.md](WORKLOG.md) |
| Quy trình làm việc (branch, commit, thí nghiệm) | [docs/workflow.md](docs/workflow.md) |
| Hợp đồng dữ liệu với repo game | [docs/data_contract.md](docs/data_contract.md) |
| Tích hợp model vào C++ | [docs/integration.md](docs/integration.md) |
| Lý do các quyết định kỹ thuật (ADR-001 – 015) | [docs/decisions/](docs/decisions/README.md) |
| Kết quả thí nghiệm | [reports/experiments.md](reports/experiments.md) |
| Thay đổi phía game | repo game, `docs/ai-changes/README.md` (nhánh `feat/dong`) |

## Quy ước

- Nhánh `develop` là nơi tích hợp, `main` nhận bản đã chốt. Mọi thay đổi đi qua nhánh `feat/…`, `fix/…`,
  `exp/…`, `docs/…`.
- Không commit dữ liệu, model, file export. Schema là nguồn sự thật cho feature.
- Chia dữ liệu theo trận/người chơi, không theo frame. Không đọc holdout để chỉnh model.
- Tài liệu viết bằng tiếng Việt. Luật cho AI trợ lý viết code: [AGENTS.md](AGENTS.md).
