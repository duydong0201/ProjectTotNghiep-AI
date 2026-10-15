"""Do DO CHINH XAC CUA DE XUAT: model de xuat cu dap dung luc bong trong tam voi khong?

Vi sao can chi so nay: macro-F1 thuong cho viec BAT DUOC lop hiem nhung KHONG PHAT viec
de xuat bua. Do ngay 15/10/2026 tren 7 tran: model dat macro-F1 0.671 ma 95,9% de xuat cu
dap xay ra luc bong cach bot 1806 px (nua san ben kia). Hai chi so nay khong cung chieu,
nen phai do rieng cai nao phan anh muc tieu that.

Cach dung:
    python scripts/eval_proposal_precision.py <duong_dan_joblib> [file1.csv file2.csv ...]

Mac dinh do tren 7 tran nguoi choi vs bot RF (14-15/10/2026).
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spike_ai.data import load_match  # noqa: E402
from spike_ai.features import build  # noqa: E402

LOGS = Path("F:/DoAnTotNghiep/ProjectTotNghiep/Logs/v1")
DEFAULT_FILES = [
    "2026-10-14_23-51-23_P01.csv",
    "2026-10-14_23-53-18_P01.csv",
    "2026-10-14_23-57-47_P01.csv",
    "2026-10-14_23-59-17_P01.csv",
    "2026-10-15_00-00-57_P01.csv",
    "2026-10-15_00-02-43_P01.csv",
    "2026-10-15_00-04-19_P01.csv",
]
SPIKES = ("SpikeLight", "SpikeMedium", "SpikeStrong")

# Nguong khoang cach (pixel) de coi la "bong trong tam dap".
# Do tu cac cu SpikeMedium THUC SU no trong game: 45-51 px. Dung 51 cho de dai.
#
# LUU Y: game kiem bang distancePercent = khoang_cach / chieu_cao_nhan_vat * 100 < 50,
# tuc nguong thuc la 50% chieu cao. Log khong co cot chieu cao nhan vat nen o day dung
# nguong pixel suy ra tu quan sat. Con so tuyet doi co the lech, nhung dung de SO SANH
# hai model tren CUNG tap frame thi khong anh huong.
REACH_PX = 51.0


def main() -> int:
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if model_path is None:
        print(__doc__)
        return 1
    files = sys.argv[2:] or DEFAULT_FILES

    payload = joblib.load(model_path)
    clf = payload["model"]
    classes = np.array(list(clf.classes_))

    n_frames = n_prop = n_prop_in_reach = 0
    dists = []
    for name in files:
        df, _ = load_match(LOGS / name)
        if df.empty:
            continue
        X, _ = build(df, "v1", "opponent")  # bot ML o san phai
        arg = classes[clf.predict_proba(X).argmax(axis=1)]
        is_spike = np.isin(arg, SPIKES)
        dist = np.hypot(df["ball_x"].to_numpy() - df["o_x"].to_numpy(), df["ball_y"].to_numpy() - df["o_y"].to_numpy())
        n_frames += len(df)
        n_prop += int(is_spike.sum())
        n_prop_in_reach += int((is_spike & (dist <= REACH_PX)).sum())
        dists += dist[is_spike].tolist()

    print(f"model : {model_path}")
    print(f"du lieu: {len(files)} tran, {n_frames} frame (agent = san phai)")
    print(f"  de xuat cu dap          : {n_prop} frame ({n_prop / n_frames * 100:.2f}% frame)")
    print(f"  trong do bong TRONG tam : {n_prop_in_reach}")
    if n_prop:
        print(f"  => DO CHINH XAC DE XUAT : {n_prop_in_reach / n_prop * 100:.2f}%")
        d = np.array(dists)
        print(
            f"  khoang cach khi de xuat : trung vi {np.median(d):.0f} px, "
            f"p5 {np.percentile(d, 5):.0f}, p95 {np.percentile(d, 95):.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
