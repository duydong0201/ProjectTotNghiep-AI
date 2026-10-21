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


def log_files(folder: Path) -> list[Path]:
    """Mọi file log trong thư mục (đệ quy): .csv và .csv.gz (log tự sinh được nén, xem selfplay.py)."""
    return sorted([*folder.rglob("*.csv"), *folder.rglob("*.csv.gz")])


def is_feature_dump(path: Path) -> bool:
    """File dump feature để đối chiếu C++/Python, không phải log trận."""
    return path.name.removesuffix(".gz").endswith("_feat.csv")


def load_match(path: str | Path) -> tuple[pd.DataFrame, str]:
    """Đọc một file log, trả về (DataFrame, version). Luôn có cột match_id và source."""
    path = Path(path)
    df = pd.read_csv(path)

    version = detect_version(df.columns)
    validate_columns(df, version)

    if version == "v0":
        df = rebuild_frames_v0(df)

    if "match_id" not in df.columns:
        df["match_id"] = path.name.removesuffix(".gz").removesuffix(".csv")
    df["match_id"] = df["match_id"].astype(str)
    # source = tên thư mục chứa file (vd: bot / human / v0)
    df["source"] = path.parent.name
    return df, version


def adapt_version(df: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
    """Đưa một DataFrame log từ schema `source` sang schema `target`.

    Cần vì một thư mục có thể chứa lẫn hai định dạng: game ghi v1 trước 12/10/2026 và
    ghi v2 sau đó, mà log của cùng một người chơi vẫn nằm chung một chỗ.

    Chỉ hỗ trợ v1 <-> v2, vì hai cái đó khác nhau đúng một cột. v0 có format hoàn toàn
    khác (thiếu toàn bộ trạng thái bóng) nên không chuyển được.
    """
    if source == target:
        return df
    if {source, target} != {"v1", "v2"}:
        raise ValueError(f"Không chuyển được schema {source} -> {target}")

    out = df.copy()
    if target == "v2":
        # Suy ra cột còn thiếu. Độ chính xác 100% trên 9.346 frame, xem ADR-009.
        out["rally_last_touch_kind"] = rebuild_kind_v1_to_v2(out)
        out["schema_version"] = 2
    else:
        # Hạ xuống v1: bỏ cột thừa. Không mất gì cho 21 feature của v1.
        out = out.drop(columns=["rally_last_touch_kind"])
        out["schema_version"] = 1
    return out


def load_dirs(dirs: list[str | Path], expected_version: str | None = None) -> pd.DataFrame:
    """Đọc toàn bộ *.csv (đệ quy) trong các thư mục, bỏ qua file rỗng.

    `expected_version` là schema ĐÍCH. File nào khác định dạng đó mà chuyển được
    (v1 <-> v2) thì tự chuyển, và số file đã chuyển được IN RA - không im lặng, vì trộn
    định dạng ngoài ý muốn là lỗi thật cần thấy ngay.
    """
    frames = []
    adapted: dict[str, int] = {}
    for d in dirs:
        d = resolve(d)
        if not d.exists():
            raise FileNotFoundError(f"Không thấy thư mục dữ liệu: {d}")
        for path in log_files(d):
            if is_feature_dump(path):
                continue  # file dump feature để đối chiếu, không phải log trận
            df, version = load_match(path)
            if df.empty:
                continue
            if expected_version and version != expected_version:
                try:
                    df = adapt_version(df, version, expected_version)
                except ValueError as exc:
                    raise ValueError(f"{path.name} là {version}, config yêu cầu {expected_version}") from exc
                adapted[version] = adapted.get(version, 0) + 1
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"Không có file CSV nào trong {dirs}")

    for version, count in sorted(adapted.items()):
        print(f"  đã chuyển {count} file từ schema {version} sang {expected_version}")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Dung lai cot rally_last_touch_kind cho log v1 (nang v1 -> v2)
# ---------------------------------------------------------------------------
# Cot nay chi duoc game ghi tu 12/10/2026. 150 tran nguoi choi thu ngay 10-10 va 10-11
# nam truoc moc do, nen khong co cot that. Nhung dung lai duoc tu thong tin DA CO trong
# log v1: rally_last_touch (AI cham) + rally_touch_count (lan thu may) + cot intent.
#
# DO CHINH XAC DA DO: 100,00% tren 9.346 frame cua 2 file v2 that (file co ca cot thuc te
# de doi chieu). Xem docs/decisions/adr-009-derive-touch-kind-for-v1-logs.md.
#
# LUU Y VE NGUON GOC: cot nay la SUY RA, khong phai do game ghi truc tiep. File nang cap
# phai de o thu muc rieng co ten noi ro dieu do (vd human_kind_derived) - `source` do
# load_match gan bang ten thu muc se theo du lieu vao luc train.
ENTITY_NONE = 8  # GameConfig::NONE, KHONG phai -1

# FinalIntent (cot p_input_intent/o_input_intent) -> ActionState (cot rally_last_touch_kind).
# HAI ENUM KHAC NHAU, vd Set la 7 ben FinalIntent nhung 6 ben ActionState.
# Jump (FinalIntent 1) khong co trong bang: no khong cham bong nen khong bao gio
# duoc ghi vao lastTouchKind.
FINAL_INTENT_TO_ACTION_STATE = {
    2: 2,  # Spike
    5: 5,  # Bump
    6: 3,  # Slide
    7: 6,  # Set
    8: 4,  # Serve
    9: 7,  # SpikeLight
    10: 8,  # SpikeMedium
    11: 9,  # SpikeStrong
}

# Khi phat hien mot cu cham moi, nhin lui toi da ngan nay frame de tim intent sinh ra no.
# Can nhin lui vi DatasetLogger ghi dong log TRUOC khi intent duoc ap dung, nen intent va
# hau qua cua no co the nam o hai frame khac nhau.
KIND_LOOKBACK = 12


def rebuild_kind_v1_to_v2(df: pd.DataFrame, lookback: int = KIND_LOOKBACK) -> pd.Series:
    """Suy ra cot rally_last_touch_kind cho mot DataFrame log v1."""
    import numpy as np

    last_touch = df["rally_last_touch"].fillna(ENTITY_NONE).astype(int).to_numpy()
    touch_count = df["rally_touch_count"].fillna(0).astype(int).to_numpy()
    player_intent = df["p_input_intent"].fillna(0).astype(int).to_numpy()
    opp_intent = df["o_input_intent"].fillna(0).astype(int).to_numpy()

    out = np.zeros(len(df), dtype=int)
    current = 0
    for k in range(len(df)):
        if last_touch[k] == ENTITY_NONE:
            # Pha bong vua reset -> RallyState{} dat lastTouchKind ve None.
            current = 0
        elif k == 0 or last_touch[k] != last_touch[k - 1] or touch_count[k] != touch_count[k - 1]:
            # Co cu cham moi. `k == 0` la bat buoc: dong dau file thuong DA co san mot cu
            # cham (cu giao bong xay ra truoc dong log dau tien), khong co dong truoc de so.
            # Thieu nhanh nay thi mat ca pha bong dau - do duoc 73/5385 frame sai.
            #
            # `touch_count` doi cung tinh la cu cham moi: khi CUNG mot nhan vat cham hai lan
            # thi lastTouch khong doi, chi touchCount tang.
            source = player_intent if last_touch[k] == 0 else opp_intent
            for j in range(k, max(-1, k - lookback), -1):
                if source[j] in FINAL_INTENT_TO_ACTION_STATE:
                    current = FINAL_INTENT_TO_ACTION_STATE[source[j]]
                    break
        out[k] = current

    return pd.Series(out, index=df.index, name="rally_last_touch_kind")
