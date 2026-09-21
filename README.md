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
- Hợp đồng dữ liệu với repo game: [docs/data_contract.md](docs/data_contract.md)
- Cách tích hợp model vào C++: [docs/integration.md](docs/integration.md)
- Nhật ký thí nghiệm: [reports/experiments.md](reports/experiments.md)

---

## 1. Cài đặt

Yêu cầu: Python 3.10+ (máy hiện có 3.11).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell / cmd)
pip install -r requirements.txt
pip install -e .                # để import được package spike_ai
```

## 2. Chạy thử pipeline (vòng mỏng end-to-end)

```bash
# 1) Copy log CSV từ repo game vào data/raw/v0
python scripts/import_logs.py --src ../ProjectTotNghiep/Logs --dest data/raw/v0

# 2) Xem nhanh dữ liệu: số trận, số dòng, phân bố nhãn
python scripts/inspect_data.py --config configs/baseline_v0.yaml

# 3) Train các model baseline, lưu vào models/, ghi kết quả vào reports/experiments.md
python -m spike_ai.train --config configs/baseline_v0.yaml

# 4) Export model Decision Tree sang header C++ để cắm vào game
python -m spike_ai.export_cpp --model models/<run>/move__decision_tree.joblib --out exports/

# 5) Chạy unit test
pytest
```

> ⚠️ Log **v0** (format CSV hiện tại của game) **không có thông tin về bóng** nên model học từ nó
> rất yếu. v0 chỉ dùng để chạy thông đường ống. Dữ liệu thật để học là **v1** — xem
> [docs/data_contract.md](docs/data_contract.md).

## 3. Cấu trúc thư mục

```text
ProjectTotNghiep-AI/
├── PLAN.md                  # kế hoạch thực hiện theo giai đoạn
├── configs/                 # cấu hình mỗi lần train (YAML) -> tái lập được thí nghiệm
├── schema/                  # NGUỒN SỰ THẬT: cột log, enum, danh sách + thứ tự feature
│   ├── legacy_v0.json       #   format CSV cũ của game
│   └── feature_spec.v1.json #   format mới đề xuất cho game
├── data/                    # KHÔNG commit (đã .gitignore)
│   ├── raw/v0/              #   log CSV cũ copy từ game
│   ├── raw/v1/bot|human/    #   log v1, tách theo nguồn điều khiển
│   └── processed/           #   dữ liệu đã clean / feature (parquet, csv)
├── notebooks/               # EDA, thử nghiệm nhanh (đặt tên 01_eda.ipynb, 02_...)
├── src/spike_ai/            # code chính (import được)
│   ├── schema.py            #   đọc spec, nhận diện version, kiểm tra cột
│   ├── data.py              #   đọc nhiều file CSV, gắn match_id
│   ├── features.py          #   raw -> X (feature) + y (nhãn)  ← phải tái tạo được trong C++
│   ├── split.py             #   chia train/test THEO TRẬN
│   ├── models.py            #   danh sách model (majority, tree, forest, knn, ...)
│   ├── evaluate.py          #   macro-F1, confusion matrix
│   ├── train.py             #   CLI train
│   └── export_cpp.py        #   CLI export model -> header C++ + golden test
├── scripts/                 # tiện ích chạy tay: import log, xem dữ liệu
├── models/                  # model đã train (.joblib) — không commit, phát hành qua Releases
├── exports/                 # file .h sinh ra để copy sang repo game
├── reports/                 # experiments.md, hình ảnh cho báo cáo
├── docs/                    # data contract, hướng dẫn tích hợp
└── tests/                   # pytest
```

## 4. Quy tắc làm việc

1. **Không commit dữ liệu và model** vào git. Dữ liệu để ở Google Drive / Git LFS,
   model phát hành qua GitHub Releases.
2. **Mọi thay đổi feature phải sửa trong `schema/*.json` trước**, rồi mới sửa `features.py`.
   Test sẽ báo lỗi nếu thứ tự feature trong code khác với spec.
3. **Chia train/test theo trận** (`match_id`), không bao giờ chia theo frame.
4. Mỗi lần train là một file config trong `configs/` → kết quả tự ghi vào `reports/experiments.md`.
