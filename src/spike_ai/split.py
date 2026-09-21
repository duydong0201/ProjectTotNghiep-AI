"""Chia dữ liệu thành train / dev / holdout bằng HASH của khoá nhóm, và cross-validation theo nhóm.

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

Khi có ít nhóm, một tập dev duy nhất rất nhiễu (một trận dài có thể chiếm phần lớn tập).
Khi đó dùng cross-validation (kfold, lopo), xem ADR-004.
"""

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

TRAIN, DEV, HOLDOUT = "train", "dev", "holdout"
SPLITS = (TRAIN, DEV, HOLDOUT)

# Một nhóm chiếm quá tỉ lệ này số frame của một tập -> kết quả tập đó chủ yếu phản ánh một nhóm
MAX_GROUP_SHARE = 0.5


def hash_unit(key: str, salt: str) -> float:
    """Ánh xạ một khoá sang số thực trong [0, 1), tất định theo (salt, key)."""
    digest = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(digest[:15], 16) / 16**15


def assign_split(key: str, salt: str, dev_ratio: float, holdout_ratio: float) -> str:
    """Tập của một khoá. Phụ thuộc DUY NHẤT vào (salt, key, tỉ lệ) - không vào khoá khác.

    Holdout chỉ phụ thuộc holdout_ratio (u < holdout_ratio), nên đổi dev_ratio
    không bao giờ làm nhóm nào ra/vào holdout.
    """
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

    match   mỗi trận là một nhóm (mặc định)
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


def split_report(groups: pd.Series, index: dict, targets: dict[str, pd.Series]) -> tuple[dict, list[str]]:
    """Mô tả độ cân bằng của từng tập và trả về danh sách cảnh báo.

    targets: {tên nhãn: Series nhãn} để đếm số mẫu mỗi lớp trong từng tập.
    Cảnh báo khi: một nhóm chiếm > MAX_GROUP_SHARE số frame của một tập,
    hoặc một lớp có trong train nhưng vắng mặt ở tập đo.
    """
    total = sum(len(idx) for idx in index.values())
    report, warnings = {}, []
    for name, idx in index.items():
        if len(idx) == 0:
            continue
        sizes = groups.iloc[idx].value_counts()
        report[name] = {
            "groups": int(len(sizes)),
            "frames": int(len(idx)),
            "frame_share": round(len(idx) / total, 4),
            "largest_group": str(sizes.index[0]),
            "largest_group_share": round(float(sizes.iloc[0] / len(idx)), 4),
            "labels": {t: {str(k): int(v) for k, v in y.iloc[idx].value_counts().items()} for t, y in targets.items()},
        }
        if name != TRAIN and report[name]["largest_group_share"] > MAX_GROUP_SHARE:
            warnings.append(
                f"{name}: nhóm '{sizes.index[0]}' chiếm {report[name]['largest_group_share']:.0%} số frame "
                f"-> kết quả chủ yếu phản ánh một nhóm"
            )

    if TRAIN in report:
        for name in report:
            if name == TRAIN:
                continue
            for t in targets:
                missing = sorted(set(report[TRAIN]["labels"][t]) - set(report[name]["labels"][t]))
                if missing:
                    warnings.append(f"{name}: nhãn '{t}' thiếu lớp {missing} (có trong train)")
    return report, warnings


def cv_folds(cv_groups: pd.Series, mode: str, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Danh sách (train_idx, val_idx) cho cross-validation theo nhóm.

    kfold  GroupKFold: chia các nhóm thành n_folds phần có tổng số frame xấp xỉ nhau
    lopo   Leave-One-Group-Out: mỗi fold bỏ ra đúng một nhóm (dùng với nhóm = người chơi)
    """
    n_groups = cv_groups.nunique()
    dummy = np.zeros(len(cv_groups))
    if mode == "kfold":
        if n_groups < n_folds:
            raise ValueError(f"kfold cần ít nhất {n_folds} nhóm, chỉ có {n_groups}. Giảm n_folds.")
        return list(GroupKFold(n_splits=n_folds).split(dummy, groups=cv_groups))
    if mode == "lopo":
        if n_groups < 2:
            raise ValueError(f"lopo cần ít nhất 2 nhóm, chỉ có {n_groups}.")
        return list(LeaveOneGroupOut().split(dummy, groups=cv_groups))
    raise ValueError(f"mode phải là 'kfold' hoặc 'lopo', nhận: {mode}")


def assert_no_overlap(train_keys, holdout_keys, what: str = "match_id") -> None:
    """Holdout nạp từ thư mục riêng không được trùng trận nào với dữ liệu train."""
    overlap = sorted(set(train_keys) & set(holdout_keys))
    if overlap:
        raise ValueError(f"Rò rỉ dữ liệu: {len(overlap)} {what} vừa ở train vừa ở holdout, ví dụ {overlap[:3]}")
