"""Doi chieu feature do C++ tinh voi feature do Python tinh, tren CUNG frame.

Vi sao can rieng buoc nay: golden test trong header model CHI kiem MODEL (cung vector vao
-> cung xac suat ra), KHONG kiem cach tinh FEATURE. Lech feature thi model van chay ma doan
sai am tham, khong co loi nao bao ra. Day la lo hong da tung ton tai truoc 09/10/2026.

Cach dung:
    1. Bat DatasetLogConfig::DUMP_ML_FEATURES = true, build lai game
    2. Chay game vai pha bong -> sinh ra <match_id>.csv va <match_id>_feat.csv
    3. python scripts/verify_features_v2.py <thu_muc_log>

File <match_id>_feat.csv do DatasetLogger ghi, moi frame 2 dong (agent=0 PLAYER va
agent=3 OPPONENT_1), nen kiem luon duoc phep lat san qua luoi.

Ket qua mong doi: 0/27 feature lech, cho CA HAI agent.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spike_ai.data import load_match  # noqa: E402
from spike_ai.features import build  # noqa: E402
from spike_ai.schema import feature_names, load_spec  # noqa: E402

# Nguong so sanh. C++ tinh bang float32 tu trang thai game, Python tinh lai tu so doc
# trong CSV, nen luon co chenh lech o muc lam tron - khong phai sai khac logic.
#
# ATOL = 1e-3 (nguong dau tien toi dat) QUA CHAT: do ngay 14/10/2026 co dung 1/5354 frame
# vuot nguong, voi sai so 0.0045 tren gia tri 28.69 (tuc 1.6e-4 tuong doi). Nguyen nhan la
# logger ghi 6 chu so co nghia nen ball_x ~1909.98 mat toi 0.005, sai so do truyen vao moi
# feature tinh bang phep tru. Da nang logger len setprecision(9), nhung van de nguong co
# bien do an toan vi feature dang hieu cua hai so ~2000 co the mat ~2 don vi cuoi float32.
#
# 0.01 van du chat de bat loi LOGIC: lech logic (sai dau, sai cot, quen lat san) tao ra
# chenh lech hang don vi tro len, khong phai 0.01.
ATOL = 1e-2
RTOL = 1e-4

AGENT_OF_ENTITY = {0: "player", 3: "opponent"}


def verify_one(log_path: Path, feat_path: Path, names: list[str], varied: set[str]) -> int:
    df, version = load_match(log_path)
    if version != "v2":
        print(f"  BO QUA {log_path.name}: la schema {version}, can v2")
        return 0

    feat = pd.read_csv(feat_path)
    n_bad = 0
    for entity, agent in AGENT_OF_ENTITY.items():
        cpp = feat[feat["agent"] == entity].set_index("frame")
        if cpp.empty:
            print(f"  {log_path.name} agent={agent}: khong co dong nao trong file _feat")
            n_bad += 1
            continue

        X, _ = build(df, "v2", agent)
        X.index = df["frame"].to_numpy()
        common = X.index.intersection(cpp.index)
        # frame trung lap (vd log bi ghi hai lan) se lam so sanh vo nghia
        if X.index.has_duplicates or cpp.index.has_duplicates:
            print(f"  {log_path.name} agent={agent}: CO FRAME TRUNG LAP, bo qua")
            n_bad += 1
            continue

        py = X.loc[common]
        c = cpp.loc[common]
        print(f"  {log_path.name} agent={agent:<9} {len(common)} frame")
        for i, name in enumerate(names):
            a = py[name].to_numpy(dtype=np.float64)
            b = c[f"f{i}"].to_numpy(dtype=np.float64)
            # Feature khong doi gia tri thi viec "khop" khong chung minh gi: ca hai ben deu
            # dang tra ve cung mot hang so. Chi tinh la DA KIEM khi no thuc su bien thien o
            # it nhat mot agent / mot tran (gop lai qua `varied`).
            if np.ptp(a) > 0.0 or np.ptp(b) > 0.0:
                varied.add(name)
            if np.allclose(a, b, atol=ATOL, rtol=RTOL):
                continue
            n_bad += 1
            d = np.abs(a - b)
            k = int(np.argmax(d))
            print(
                f"      LECH f{i:<2} {name:<20} lech toi da {d.max():.6g} "
                f"o frame {common[k]}: python={a[k]:.6g} cpp={b[k]:.6g} "
                f"({int((d > ATOL).sum())}/{len(d)} frame lech)"
            )
    return n_bad


def main() -> int:
    log_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "F:/DoAnTotNghiep/ProjectTotNghiep/Logs/v2")
    names = feature_names(load_spec("v2"))
    print(f"Doi chieu {len(names)} feature cua schema v2 trong {log_dir}\n")

    pairs = []
    for feat_path in sorted(log_dir.glob("*_feat.csv")):
        log_path = feat_path.with_name(feat_path.name.replace("_feat.csv", ".csv"))
        if log_path.exists():
            pairs.append((log_path, feat_path))

    if not pairs:
        print("Khong tim thay cap <match_id>.csv + <match_id>_feat.csv nao.")
        print("Da bat DatasetLogConfig::DUMP_ML_FEATURES = true va build lai chua?")
        return 1

    varied: set[str] = set()
    total_bad = sum(verify_one(lp, fp, names, varied) for lp, fp in pairs)
    print()
    if total_bad:
        print(f"THAT BAI: {total_bad} feature lech. KHONG duoc export model truoc khi sua.")
        return 1

    unexercised = sorted(set(names) - varied)
    checked = len(names) - len(unexercised)
    print(f"KHOP: {len(names)}/{len(names)} feature, tren {len(pairs)} tran, ca hai agent.")
    if unexercised:
        print()
        print(f"NHUNG {len(unexercised)} feature KHONG DOI GIA TRI trong du lieu nay, nen viec")
        print("khop KHONG chung minh gi - ca hai ben chi dang tra ve cung mot hang so:")
        for name in unexercised:
            print(f"    - {name}")
        print()
        print(f"=> Thuc su da kiem duoc: {checked}/{len(names)} feature.")
        print("   Chay lai tren du lieu co cac tinh huong do (vd log nguoi choi that co Set/Bump)")
        print("   thi moi dong lai duoc lo hong nay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
