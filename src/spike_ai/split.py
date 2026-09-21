"""Chia dữ liệu thành train / dev / holdout bằng HASH của khoá nhóm.

Vì sao chia theo nhóm (trận hoặc người chơi)?
    Các frame liền nhau gần như giống hệt nhau. Nếu chia theo frame, frame 100 nằm ở train
    còn frame 101 nằm ở test -> model "nhìn thấy đáp án", điểm cao ảo (data leakage).

Vì sao dùng hash thay vì random?
    - Tất định: chạy lại luôn ra cùng kết quả, không phụ thuộc seed hay thứ tự đọc file.
    - Ổn định: thêm trận mới KHÔNG làm trận cũ đổi tập. Holdout đã chốt thì giữ nguyên.
    - Không ai chọn tay được trận nào vào tập nào.

Ba tập:
    train    dùng để fit model
    dev      dùng để so sánh model, chỉnh hyperparameter, phân tích lỗi
    holdout  CHỈ ĐỂ ĐO kết quả cuối (train --final). Không được lấy lỗi trên holdout ra để
             sửa feature/model - đó là "tune vào tập kiểm chứng". Xem docs/workflow.md.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TRAIN, DEV, HOLDOUT = "train", "dev", "holdout"
SPLITS = (TRAIN, DEV, HOLDOUT)


def hash_unit(key: str, salt: str) -> float:
    """Ánh xạ một khoá sang số thực trong [0, 1), tất định theo (salt, key)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(digest[:15], 16) / 16**15


def assign_split(key: str, salt: str, dev_ratio: float, holdout_ratio: float) -> str:
    """Tập của một khoá. Phụ thuộc DUY NHẤT vào (salt, key, tỉ lệ) - không vào khoá khác."""
    if not (0 <= dev_ratio and 0 <= holdout_ratio and dev_ratio + holdout_ratio < 1):
        raise ValueError(f"Tỉ lệ không hợp lệ: dev={dev_ratio}, holdout={holdout_ratio}")
    u = hash_unit(str(key), salt)
    if u < holdout_ratio:
        return HOLDOUT
    if u < holdout_ratio + dev_ratio:
        return DEV
    return TRAIN


def group_keys(df: pd.DataFrame, group_by: str, agent: str) -> pd.Series:
    """Khoá nhóm cho mỗi dòng.

    match   mỗi trận là một nhóm (mặc định, dùng cho dữ liệu bot và v0)
    player  mỗi người chơi là một nhóm -> đo được model có tổng quát sang NGƯỜI MỚI không
    """
    if group_by == "match":
        return df["match_id"].astype(str)
    if group_by == "player":
        col = "p_player_id" if agent == "player" else "o_player_id"
        if col not in df.columns:
            raise ValueError(f"group_by=player cần cột {col} (chỉ có ở schema v1)")
        return df[col].astype(str)
    raise ValueError(f"group_by phải là 'match' hoặc 'player', nhận: {group_by}")


def split_by_hash(groups: pd.Series, salt: str, dev_ratio: float, holdout_ratio: float) -> dict:
    """Trả về {tên tập: mảng index dòng}. Báo lỗi nếu một tập có tỉ lệ > 0 mà lại rỗng."""
    mapping = {k: assign_split(k, salt, dev_ratio, holdout_ratio) for k in groups.unique()}
    labels = groups.map(mapping).to_numpy()

    result = {name: np.flatnonzero(labels == name) for name in SPLITS}
    ratios = {TRAIN: 1 - dev_ratio - holdout_ratio, DEV: dev_ratio, HOLDOUT: holdout_ratio}
    empty = [name for name in SPLITS if ratios[name] > 0 and len(result[name]) == 0]
    if empty:
        raise ValueError(
            f"Tập {empty} rỗng với {len(mapping)} nhóm. Cần thêm dữ liệu hoặc đổi tỉ lệ "
            f"(không đổi salt sau khi đã đo trên holdout)."
        )
    return result


def split_manifest(groups: pd.Series, index: dict) -> dict:
    """{tên tập: danh sách khoá nhóm} - lưu kèm mỗi lần train để biết nhóm nào nằm ở đâu."""
    return {name: sorted(groups.iloc[idx].unique().tolist()) for name, idx in index.items()}


def kfold_by_group(groups: pd.Series, n_splits: int = 5):
    """Cross-validation theo nhóm trên phần train+dev (dùng ở GĐ4)."""
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)
