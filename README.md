# ProjectTotNghiep-AI — AI học từ hành vi người chơi

Phần AI/ML của đồ án **The Spike Cross Remaster** (game bóng chuyền 2D, repo game:
[thanh551419a/ProjectTotNghiep](https://github.com/thanh551419a/ProjectTotNghiep)).

Mục tiêu: học một **policy** bắt chước hành vi người chơi (Behavior Cloning):

```text
trạng thái game tại frame t  ──►  model  ──►  hành động (move, action) tại frame t
```

Model sau khi train được export sang C/C++ và cắm vào game dưới dạng một Intent producer
(`MLInputSystem`), thay thế cho `AIInputSystem` rule-based hiện tại.

- Kế hoạch thực hiện: [PLAN.md](PLAN.md)
- **Quy trình làm việc** (branch, commit, check, thí nghiệm): [docs/workflow.md](docs/workflow.md)
- Lý do các quyết định kỹ thuật: [docs/decisions/](docs/decisions/README.md)
- Hợp đồng dữ liệu với repo game: [docs/data_contract.md](docs/data_contract.md)
- Cách tích hợp model vào C++: [docs/integration.md](docs/integration.md)
- Nhật ký thí nghiệm: [reports/experiments.md](reports/experiments.md)

---

## 1. Cài đặt

Yêu cầu: Python 3.10+ (máy hiện có 3.11).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell / cmd)
pip install -r requirements-dev.txt   # gồm requirements.txt + pytest, ruff, jupyter
pip install -e .                      # để import được package spike_ai

# kiểm tra mọi thứ hoạt động (ruff + pytest)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

## 2. Chạy thử pipeline (vòng mỏng end-to-end)

```bash
# 1) Copy log CSV từ repo game vào data/raw/v0
python scripts/import_logs.py --src ../ProjectTotNghiep/Logs --dest data/raw/v0

# 2) Xem nhanh dữ liệu: số trận, số dòng, phân bố nhãn
python scripts/inspect_data.py --config configs/baseline_v0.yaml

# 3) Train, đánh giá bằng cross-validation (v0 có ít trận), lưu vào models/, ghi vào reports/experiments.md
#    (holdout được giữ kín; chỉ thêm --final khi đã chốt model — xem docs/workflow.md, ADR-004)
python -m spike_ai.train --config configs/baseline_v0.yaml

# 4) Export model Decision Tree sang header C++ để cắm vào game
python -m spike_ai.export_cpp --model models/<run>/move__decision_tree.joblib --out exports/

# 5) Kiểm tra trước khi commit
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

> ⚠️ Log **v0** (format CSV hiện tại của game) **không có thông tin về bóng** nên model học từ nó
> rất yếu. v0 chỉ dùng để chạy thông đường ống. Dữ liệu thật để học là **v1** — xem
> [docs/data_contract.md](docs/data_contract.md).

## 3. Cấu trúc thư mục

```text
ProjectTotNghiep-AI/
├── PLAN.md                  # kế hoạch thực hiện theo giai đoạn
├── WORKLOG.md               # việc đã xong + bằng chứng
├── JOURNAL.md               # nhật ký tuần (dùng khi gặp giảng viên)
├── AGENTS.md                # luật cho AI trợ lý viết code
├── .github/workflows/ci.yml # CI: chạy scripts/check.sh trên develop/main
├── configs/                 # cấu hình mỗi lần train (YAML) -> tái lập được thí nghiệm
├── schema/                  # NGUỒN SỰ THẬT: cột log, enum, danh sách + thứ tự feature
│   ├── legacy_v0.json       #   format CSV cũ của game
│   └── feature_spec.v1.json #   format mới đề xuất cho game
├── data/                    # KHÔNG commit (đã .gitignore)
│   ├── raw/v0/              #   log CSV cũ copy từ game
│   ├── raw/v1/bot|human/    #   log v1, tách theo nguồn điều khiển
│   ├── raw/v1/human_holdout/ #  holdout tương lai: thu SAU khi chốt model
│   └── processed/           #   dữ liệu đã clean / feature (parquet, csv)
├── notebooks/               # EDA, thử nghiệm nhanh (đặt tên 01_eda.ipynb, 02_...)
├── src/spike_ai/            # code chính (import được)
│   ├── schema.py            #   đọc spec, nhận diện version, kiểm tra cột
│   ├── data.py              #   đọc nhiều file CSV, gắn match_id
│   ├── features.py          #   raw -> X (feature) + y (nhãn)  ← phải tái tạo được trong C++
│   ├── split.py             #   chia theo HASH của trận/người chơi, CV (kfold/lopo), báo cáo cân bằng
│   ├── models.py            #   danh sách model (majority, tree, forest, knn, ...)
│   ├── evaluate.py          #   macro-F1 (chỉ lớp có mặt), support, confusion matrix
│   ├── train.py             #   CLI train
│   └── export_cpp.py        #   CLI export model -> header C++ + golden test
├── scripts/                 # check.ps1/.sh (quality gate), import log, xem dữ liệu
├── models/                  # model đã train (.joblib) — không commit, phát hành qua Releases
├── exports/                 # file .h sinh ra để copy sang repo game
├── reports/                 # experiments.md; results/<ngày>/ = kết quả đã chốt cho báo cáo
├── docs/
│   ├── workflow.md          #   quy trình làm việc (nguồn duy nhất)
│   ├── data_contract.md     #   hợp đồng dữ liệu với repo game
│   ├── integration.md       #   tích hợp model vào C++
│   ├── decisions/           #   ADR
│   └── plans/               #   template.md, active/, completed/
└── tests/                   # pytest
```

## 4. Quy trình làm việc

Xem [docs/workflow.md](docs/workflow.md). Tóm tắt:

- Nhánh `develop` là nơi làm việc chính, `main` chỉ nhận bản đã chốt (gắn tag). Mọi thay đổi đi qua
  nhánh `feat/…`, `fix/…`, `exp/…` rồi mở PR vào `develop`.
- `scripts/check.ps1` phải PASS trước mỗi commit. CI chạy lại đúng các bước đó.
- Không commit dữ liệu và model. Schema là nguồn sự thật cho feature. Chia dữ liệu theo trận/người chơi.
  Holdout chỉ đo khi đã chốt model.
