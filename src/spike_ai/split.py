"""Chia train/test THEO TRẬN.

Các frame liền nhau gần như giống hệt nhau. Nếu chia ngẫu nhiên theo frame, frame 100
nằm ở train còn frame 101 nằm ở test -> model "nhìn thấy đáp án", điểm cao ảo.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def train_test_by_match(groups: pd.Series, test_size: float = 0.2, seed: int = 42):
    """Trả về (train_idx, test_idx) sao cho một trận chỉ nằm ở một phía."""
    if groups.nunique() < 2:
        raise ValueError("Cần ít nhất 2 trận để chia train/test.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def kfold_by_match(groups: pd.Series, n_splits: int = 5):
    """Cross-validation theo trận (dùng ở GĐ4)."""
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups)
