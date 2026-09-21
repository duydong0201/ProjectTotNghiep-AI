"""Đọc log CSV của game. Mỗi file = một trận (match)."""

from pathlib import Path

import pandas as pd

from .paths import resolve
from .schema import detect_version, validate_columns


def load_match(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Đọc một file log, trả về (DataFrame, version). Luôn có cột match_id và source."""
    path = Path(path)
    df = pd.read_csv(path)
    version = detect_version(df.columns)
    validate_columns(df, version)

    if "match_id" not in df.columns:
        df["match_id"] = path.stem
    df["match_id"] = df["match_id"].astype(str)
    # source = tên thư mục chứa file (vd: bot / human / v0)
    df["source"] = path.parent.name
    return df, version


def load_dirs(dirs: list[str | Path], expected_version: str | None = None) -> pd.DataFrame:
    """Đọc toàn bộ *.csv (đệ quy) trong các thư mục, bỏ qua file rỗng."""
    frames = []
    for d in dirs:
        d = resolve(d)
        if not d.exists():
            raise FileNotFoundError(f"Không thấy thư mục dữ liệu: {d}")
        for path in sorted(d.rglob("*.csv")):
            df, version = load_match(path)
            if df.empty:
                continue
            if expected_version and version != expected_version:
                raise ValueError(f"{path.name} là {version}, config yêu cầu {expected_version}")
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Không có file CSV nào trong {dirs}")

    return pd.concat(frames, ignore_index=True)
