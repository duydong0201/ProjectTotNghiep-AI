"""Đo macro-F1 thay đổi thế nào theo SỐ MẪU, trên dữ liệu tổng hợp.

    python scripts/learning_curve.py
    python scripts/learning_curve.py --dirs data/raw/v0_synthetic --target spike_choice

Trả lời câu hỏi "cần bao nhiêu mẫu thì đủ?" bằng một đường cong thay vì phỏng đoán. Vì dữ
liệu tổng hợp có quy luật ĐÃ BIẾT và trần lý thuyết tính được, đường cong này cho biết với N
mẫu thì pipeline đạt bao nhiêu phần trăm của trần.

Cách đọc: tìm N hiện có của dữ liệu thật trên trục hoành, đối chiếu sang macro-F1 và cột std.
Std lớn nghĩa là con số đo được không đáng tin, bất kể giá trị trung bình là bao nhiêu.

LƯU Ý: đây là đường cong của bài toán TỔNG HỢP. Dữ liệu người chơi khó hơn (quy luật phức tạp
hơn, nhiễu nhiều hơn, và v0 thiếu thông tin bóng), nên con số thật sẽ thấp hơn ở cùng N.
Đường cong dùng để đọc XU HƯỚNG và độ ổn định, không phải để dự đoán điểm tuyệt đối.
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from spike_ai.data import load_dirs
from spike_ai.features import build
from spike_ai.split import cv_folds, group_keys

SAMPLE_TARGETS = (40, 80, 150, 300, 450, 900, 1800, 10**9)


def run(dirs: list[str], target: str, seed: int) -> None:
    df = load_dirs(dirs, expected_version="v0")
    X, y = build(df, "v0", "player")
    groups = group_keys(df, "match", "player")

    labelled = y[target].notna().to_numpy()
    # np.asarray(dtype=object): groups là StringArray của pandas, rng.shuffle không đảm bảo đúng trên nó
    matches = np.asarray(groups[labelled].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)

    print(f"Tổng: {labelled.sum():,} mẫu có nhãn '{target}' trong {len(matches)} trận\n")
    print(f"{'N mẫu':>8}{'số trận':>9}{'macro-F1':>11}{'std':>9}   cảnh báo")

    for want in SAMPLE_TARGETS:
        # lấy dần từng trận cho tới khi đủ số mẫu mong muốn (chia theo trận, không theo frame)
        taken, n = [], 0
        for m in matches:
            if n >= want:
                break
            taken.append(m)
            n += int((groups[labelled] == m).sum())
        if len(taken) < 5:  # cần ít nhất 5 trận cho 5 fold
            continue

        keep = labelled & groups.isin(taken).to_numpy()
        idx = np.flatnonzero(keep)
        Xs, ys = X.iloc[idx].reset_index(drop=True), y[target].iloc[idx].reset_index(drop=True)
        gs = groups.iloc[idx].reset_index(drop=True)

        scores = []
        for tr, va in cv_folds(gs, "kfold", 5):
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ).fit(Xs.iloc[tr], ys.iloc[tr])
            scores.append(f1_score(ys.iloc[va], model.predict(Xs.iloc[va]), average="macro", zero_division=0))

        s = np.array(scores)
        warn = "std > 0.05 -> chưa đáng tin" if s.std() > 0.05 else ""
        print(f"{len(idx):>8}{len(taken):>9}{s.mean():>11.3f}{s.std():>9.3f}   {warn}")
        if want == SAMPLE_TARGETS[-1]:
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", default=["data/raw/v0_synthetic"])
    parser.add_argument("--target", default="spike_choice")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.dirs, args.target, args.seed)


if __name__ == "__main__":
    main()
