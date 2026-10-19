"""So sanh model tai CHINH nhung trang thai bot ML that su roi vao trong tran.

Day la phep do DAgger can: DAgger sua lech phan bo (model hoc tu tinh huong nguoi gap,
nhung bot lai roi vao tinh huong khac), nen phai do tren trang thai CUA BOT, khong phai
tren tap kiem tra lay tu du lieu nguoi choi.

Du lieu: cac tran nguoi choi vs bot ML (san phai), chi lay frame san phai do BOT dieu khien
(o_controller = 2) - tuc dung nhung trang thai bot tu dua minh vao khi duoc trien khai.

Hai chi so cho moi loai hanh dong:
  - % frame model chon hanh dong do (argmax): model co CHIU de xuat khong
  - do chinh xac de xuat: trong nhung lan de xuat, bao nhieu lan bong that su trong vung
    detect cua hanh dong do (chi do duoc cho hanh dong cham bong)

Nguong vung detect (pixel) suy tu quan sat: cu SpikeMedium that su no o 45-51 px, ung voi
50% chieu cao nhan vat -> chieu cao ~102 px. Game kiem Spike* < 50%, Bump <= 60%, Set <= 100%.
Log khong co cot chieu cao nhan vat nen so tuyet doi co the lech, nhung SO SANH hai model
tren cung tap frame thi khong anh huong.

Cach dung:
    python scripts/eval_on_bot_states.py <model_a.joblib> <model_b.joblib> [file.csv ...]
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spike_ai.data import load_match  # noqa: E402
from spike_ai.features import build  # noqa: E402

LOGS = Path("F:/DoAnTotNghiep/ProjectTotNghiep/Logs/v1")
# 5 tran nguoi choi that vs bot ML (RF khong weight), ngay 15/10/2026
DEFAULT_FILES = [
    "2026-10-15_00-25-05_P01.csv",
    "2026-10-15_00-26-31_P01.csv",
    "2026-10-15_17-19-18_P01.csv",
    "2026-10-15_17-21-12_P01.csv",
    "2026-10-15_17-23-15_P01.csv",
]
BOT = 2
HEIGHT_PX = 102.0
REACH_PX = {
    "SpikeLight": 0.50 * HEIGHT_PX,
    "SpikeMedium": 0.50 * HEIGHT_PX,
    "SpikeStrong": 0.50 * HEIGHT_PX,
    "Spike": 0.50 * HEIGHT_PX,
    "Bump": 0.60 * HEIGHT_PX,
    "Set": 1.00 * HEIGHT_PX,
}
SHOW = ["Set", "Jump", "Bump", "SpikeLight", "SpikeMedium", "SpikeStrong", "Serve"]


def load_bot_states(files):
    Xs, dists = [], []
    for name in files:
        df, _ = load_match(LOGS / name)
        if df.empty:
            continue
        X, _ = build(df, "v1", "opponent")
        bot = (df["o_controller"] == BOT).to_numpy()
        dist = np.hypot(df["ball_x"].to_numpy() - df["o_x"].to_numpy(), df["ball_y"].to_numpy() - df["o_y"].to_numpy())
        Xs.append(X[bot])
        dists.append(dist[bot])
    import pandas as pd

    return pd.concat(Xs, ignore_index=True), np.concatenate(dists)


def describe(model_path, X, dist):
    payload = joblib.load(model_path)
    clf = payload["model"]
    classes = np.array(list(clf.classes_))
    arg = classes[clf.predict_proba(X).argmax(axis=1)]
    out = {}
    for c in SHOW:
        mask = arg == c
        n = int(mask.sum())
        prec = None
        if c in REACH_PX and n:
            prec = float((dist[mask] <= REACH_PX[c]).mean())
        out[c] = (n, n / len(X), prec)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    files = sys.argv[3:] or DEFAULT_FILES
    X, dist = load_bot_states(files)
    print(f"{len(files)} tran, {len(X)} frame san phai do BOT dieu khien\n")
    ra, rb = describe(a, X, dist), describe(b, X, dist)
    print(f"A = {a.parent.name}")
    print(f"B = {b.parent.name}\n")
    print(f"{'hanh dong':<13}{'A: so frame':>12}{'A: chinh xac':>14}{'B: so frame':>13}{'B: chinh xac':>14}")
    for c in SHOW:
        na, _, pa = ra[c]
        nb, _, pb = rb[c]
        fa = "-" if pa is None else f"{pa * 100:.1f}%"
        fb = "-" if pb is None else f"{pb * 100:.1f}%"
        print(f"{c:<13}{na:>12}{fa:>14}{nb:>13}{fb:>14}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
