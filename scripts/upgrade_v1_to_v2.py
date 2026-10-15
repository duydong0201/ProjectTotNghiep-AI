"""Nang log v1 len v2 bang cach SUY RA cot rally_last_touch_kind.

Vi sao can: cot nay chi duoc game ghi tu 12/10/2026 (xem DatasetLogger.cpp). 150 tran
nguoi choi thu ngay 10-10 va 10-11 nam truoc moc do. Chơi lai het thi ton thoi gian, ma
thong tin can thiet DA CO trong log v1: rally_last_touch + rally_touch_count + cot intent.

DO CHINH XAC: 100,00% tren 9.346 frame cua 2 file v2 that. Script tu do lai moi lan chay
(tham so --verify) nen khong phai tin vao con so cu.

QUAN TRONG VE NGUON GOC DU LIEU: cot sinh ra la SUY RA, khong phai do game ghi truc tiep.
Script BUOC thu muc dich phai co ten chua "derive" de dieu nay khong bi quen - cot `source`
ma load_match gan bang ten thu muc se di theo du lieu vao luc train.

Cach dung:
    # tu kiem chung tren file v2 that truoc
    python scripts/upgrade_v1_to_v2.py --verify <thu_muc_co_log_v2_that>

    # roi nang cap
    python scripts/upgrade_v1_to_v2.py <thu_muc_v1> <thu_muc_dich_co_chu_derive>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spike_ai.data import rebuild_kind_v1_to_v2  # noqa: E402
from spike_ai.schema import detect_version  # noqa: E402

KIND_COLUMN = "rally_last_touch_kind"


def verify(log_dir: Path) -> int:
    """Do do chinh xac tren cac file DA CO cot that."""
    files = [p for p in sorted(log_dir.rglob("*.csv")) if not p.name.endswith("_feat.csv")]
    checked = 0
    total_frames = 0
    total_ok = 0
    for path in files:
        # Log cua tran bi tat giua chung co the rong hoan toan (chua kip ghi header).
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty or KIND_COLUMN not in df.columns:
            continue
        truth = df[KIND_COLUMN].astype(int)
        pred = rebuild_kind_v1_to_v2(df)
        ok = truth == pred
        checked += 1
        total_frames += len(df)
        total_ok += int(ok.sum())
        flag = "" if ok.all() else "   <-- CO SAI LECH"
        print(f"  {path.name:<42} {ok.mean() * 100:6.2f}%  ({len(df)} frame){flag}")
        if not ok.all():
            pairs = pd.DataFrame({"that": truth[~ok], "du_doan": pred[~ok]})
            for (a, b), n in pairs.value_counts().head(5).items():
                print(f"       ActionState that={a} du_doan={b}: {n} frame")

    if not checked:
        print(f"Khong co file nao trong {log_dir} co cot {KIND_COLUMN} de doi chieu.")
        return 1
    print(f"\nTONG: {total_ok}/{total_frames} frame khop ({total_ok / total_frames * 100:.2f}%) tren {checked} file.")
    return 0 if total_ok == total_frames else 1


def upgrade(src_dir: Path, dst_dir: Path) -> int:
    if "derive" not in dst_dir.name.lower():
        print(f"TU CHOI: ten thu muc dich phai chua 'derive' - '{dst_dir.name}' thi khong.")
        print("Cot rally_last_touch_kind o day la SUY RA, khong phai do game ghi. Ten thu muc")
        print("la cho duy nhat thong tin do di theo du lieu (load_match gan no vao cot `source`).")
        return 1

    files = [p for p in sorted(src_dir.rglob("*.csv")) if not p.name.endswith("_feat.csv")]
    if not files:
        print(f"Khong co file .csv nao trong {src_dir}")
        return 1

    dst_dir.mkdir(parents=True, exist_ok=True)
    written = skipped_existing = skipped_not_v1 = 0
    for path in files:
        out_path = dst_dir / path.name
        if out_path.exists():
            skipped_existing += 1
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        try:
            version = detect_version(df.columns)
        except ValueError as exc:
            print(f"  BO QUA {path.name}: {exc}")
            skipped_not_v1 += 1
            continue
        if version != "v1":
            skipped_not_v1 += 1
            continue

        df[KIND_COLUMN] = rebuild_kind_v1_to_v2(df)
        # schema_version phai thanh 2, neu khong validate_columns se bao loi
        df["schema_version"] = 2
        df.to_csv(out_path, index=False)
        written += 1

    print(f"Da ghi {written} file v2 vao {dst_dir}")
    if skipped_existing:
        print(f"  bo qua {skipped_existing} file da ton tai (khong ghi de)")
    if skipped_not_v1:
        print(f"  bo qua {skipped_not_v1} file khong phai v1")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="thu muc chua log")
    ap.add_argument("dst", type=Path, nargs="?", help="thu muc dich (ten phai chua 'derive')")
    ap.add_argument("--verify", action="store_true", help="chi DO do chinh xac tren file da co cot that, khong ghi gi")
    args = ap.parse_args()

    if args.verify:
        print(f"Doi chieu cot {KIND_COLUMN} suy ra vs cot that, trong {args.src}\n")
        return verify(args.src)
    if args.dst is None:
        ap.error("can thu muc dich, hoac dung --verify")
    return upgrade(args.src, args.dst)


if __name__ == "__main__":
    raise SystemExit(main())
