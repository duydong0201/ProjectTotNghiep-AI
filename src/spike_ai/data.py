"""Đọc log CSV của game. Mỗi file = một trận (match)."""

from pathlib import Path

import pandas as pd

from .paths import resolve
from .schema import detect_version, load_spec, validate_columns

NO_EVENT = "None"
POS_COLUMNS = ["PlayerPosX", "PlayerPosY", "BotPosX", "BotPosY"]

# Lỗ hổng dài hơn ngần này (frame) coi như nghỉ giữa hai pha bóng - reset điểm, nhân vật bị
# đặt lại vị trí - nên không điền. Lỗ hổng ngắn hơn là lúc cả hai thật sự đứng yên.
MAX_FILL_GAP = 60


def _first_real_event(events: pd.Series) -> str:
    real = events[events != NO_EVENT]
    return real.iloc[0] if len(real) else NO_EVENT


def rebuild_frames_v0(df: pd.DataFrame, max_fill_gap: int = MAX_FILL_GAP) -> pd.DataFrame:
    """Dựng lại log v0 thành đúng một dòng cho mỗi frame.

    Game ghi log theo sự kiện chứ không theo frame: `ApplyHorizontalMovement` ghi một dòng
    cho mỗi nhân vật CÓ intent, và trong dòng đó nhân vật còn lại luôn bị ghi là "None".
    Hệ quả là log thô có ba lỗi làm hỏng nhãn `move`:

    1. Cả hai cùng di chuyển -> hai dòng cùng số frame, mỗi dòng ghi sai nhân vật kia
       thành "None" (17% số frame).
    2. Frame không ai bấm phím thì không có dòng nào -> nhãn "đứng yên" gần như biến mất
       (45% số frame trống).
    3. Dòng ghi sau đã phản ánh vị trí vừa cập nhật của nhân vật ghi trước.

    Cách sửa: gộp các dòng cùng frame (giữ event khác "None" của từng nhân vật, lấy vị trí
    ở dòng cuối cùng - lúc cả hai đã di chuyển xong), rồi điền frame trống bằng vị trí giữ
    nguyên và event "None".
    """
    if df.empty:
        return df

    known_events = set(load_spec("v0")["event_values"])
    out = df.dropna(subset=["Frame", *POS_COLUMNS]).copy()
    out["Frame"] = out["Frame"].astype(int)
    for col in ("PlayerEvent", "BotEvent"):
        # Dòng bị ghi dở (file bị cắt giữa chừng) cho ra giá trị lạ hoặc NaN -> coi như không có event
        out[col] = out[col].where(out[col].isin(known_events), NO_EVENT)

    merged = out.groupby("Frame", sort=True).agg(
        PlayerPosX=("PlayerPosX", "last"),
        PlayerPosY=("PlayerPosY", "last"),
        PlayerEvent=("PlayerEvent", _first_real_event),
        BotPosX=("BotPosX", "last"),
        BotPosY=("BotPosY", "last"),
        BotEvent=("BotEvent", _first_real_event),
    )

    logged = merged.index.to_numpy()
    wanted = [int(logged[0])]
    for prev, cur in zip(logged, logged[1:], strict=False):
        if 0 < cur - prev - 1 <= max_fill_gap:
            wanted.extend(range(int(prev) + 1, int(cur)))
        wanted.append(int(cur))

    filled = merged.reindex(wanted)
    filled[POS_COLUMNS] = filled[POS_COLUMNS].ffill()
    filled[["PlayerEvent", "BotEvent"]] = filled[["PlayerEvent", "BotEvent"]].fillna(NO_EVENT)
    return filled.reset_index()


def load_match(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Đọc một file log, trả về (DataFrame, version). Luôn có cột match_id và source."""
    path = Path(path)
    df = pd.read_csv(path)
    version = detect_version(df.columns)
    validate_columns(df, version)
    if version == "v0":
        df = rebuild_frames_v0(df)

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
