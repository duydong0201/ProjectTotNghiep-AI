"""Test chặn rò rỉ dữ liệu giữa train / dev / holdout.

Các test này là "hàng rào": nếu ai đó (kể cả chính bạn vài tuần sau) sửa cách chia làm
cho holdout bị lộ vào train, hoặc đổi salt, test sẽ đỏ.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spike_ai.split import (
    DEV,
    HOLDOUT,
    SPLITS,
    TRAIN,
    assert_no_overlap,
    assign_split,
    cv_folds,
    group_keys,
    split_by_hash,
    split_report,
)

SALT = "spike-ai-split-v1"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _groups(n_matches: int, frames: int = 5) -> pd.Series:
    return pd.Series(np.repeat([f"match_{i:03d}" for i in range(n_matches)], frames))


# ---------------------------------------------------------------- tính chất của hàm chia
def test_splits_are_disjoint_by_group():
    groups = _groups(60)
    index = split_by_hash(groups, SALT, 0.2, 0.2)
    keys = {name: set(groups.iloc[idx]) for name, idx in index.items()}
    assert keys[TRAIN].isdisjoint(keys[DEV])
    assert keys[TRAIN].isdisjoint(keys[HOLDOUT])
    assert keys[DEV].isdisjoint(keys[HOLDOUT])
    # mọi dòng thuộc đúng một tập
    assert sorted(np.concatenate(list(index.values()))) == list(range(len(groups)))


def test_split_is_deterministic():
    groups = _groups(40)
    a = split_by_hash(groups, SALT, 0.2, 0.2)
    b = split_by_hash(groups.sample(frac=1, random_state=1).sort_index(), SALT, 0.2, 0.2)
    for name in SPLITS:
        np.testing.assert_array_equal(a[name], b[name])


def test_adding_matches_never_moves_existing_ones():
    """Thêm dữ liệu mới không được làm trận cũ đổi tập - holdout đã chốt phải giữ nguyên."""
    before = {k: assign_split(k, SALT, 0.2, 0.2) for k in [f"match_{i:03d}" for i in range(30)]}
    groups = _groups(80)
    index = split_by_hash(groups, SALT, 0.2, 0.2)
    after = {k: name for name, idx in index.items() for k in groups.iloc[idx].unique()}
    assert all(after[k] == v for k, v in before.items())


def test_ratios_are_roughly_respected():
    labels = [assign_split(f"m{i}", SALT, 0.2, 0.2) for i in range(5000)]
    share = {name: labels.count(name) / len(labels) for name in SPLITS}
    assert share[HOLDOUT] == pytest.approx(0.2, abs=0.03)
    assert share[DEV] == pytest.approx(0.2, abs=0.03)


def test_empty_split_is_an_error():
    with pytest.raises(ValueError, match="rỗng"):
        split_by_hash(_groups(1), SALT, 0.2, 0.2)


@pytest.mark.parametrize("dev,holdout", [(-0.1, 0.2), (0.5, 0.5), (0.7, 0.4)])
def test_invalid_ratios(dev, holdout):
    with pytest.raises(ValueError):
        assign_split("m", SALT, dev, holdout)


def test_group_by_player_keeps_each_player_in_one_split():
    df = pd.DataFrame(
        {
            "match_id": [f"m{i}" for i in range(40)],
            "p_player_id": [f"P{i % 10:02d}" for i in range(40)],
        }
    )
    groups = group_keys(df, "player", agent="player")
    index = split_by_hash(groups, SALT, 0.2, 0.2)
    owner = {}
    for name, idx in index.items():
        for player in groups.iloc[idx]:
            assert owner.setdefault(player, name) == name


def test_group_by_player_requires_v1_column():
    with pytest.raises(ValueError, match="p_player_id"):
        group_keys(pd.DataFrame({"match_id": ["m"]}), "player", agent="player")


# ---------------------------------------------------------------- hàng rào cho config
def test_all_configs_share_the_frozen_salt():
    """Đổi salt = đổi toàn bộ cách chia = holdout cũ bị lộ. Muốn đổi phải sửa test này có chủ ý."""
    for path in CONFIG_DIR.glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert cfg["split"]["salt"] == SALT, path.name


def test_zero_holdout_ratio_is_allowed():
    index = split_by_hash(_groups(20), SALT, 0.2, 0.0)
    assert len(index[HOLDOUT]) == 0 and len(index[DEV]) > 0


def test_changing_dev_ratio_never_moves_groups_in_or_out_of_holdout():
    keys = [f"match_{i:03d}" for i in range(200)]
    a = {k for k in keys if assign_split(k, SALT, 0.1, 0.2) == HOLDOUT}
    b = {k for k in keys if assign_split(k, SALT, 0.3, 0.2) == HOLDOUT}
    assert a == b


# ---------------------------------------------------------------- báo cáo cân bằng
def test_split_report_warns_about_dominant_group_and_missing_labels():
    groups = pd.Series(["a"] * 90 + ["b"] * 10 + ["c"] * 50 + ["d"] * 50)
    labels = pd.Series(["None"] * 100 + ["None"] * 45 + ["Spike"] * 5 + ["None"] * 50)
    index = {DEV: np.arange(100), TRAIN: np.arange(100, 200)}
    report, warnings = split_report(groups, index, {"action": labels})

    assert report[DEV]["largest_group"] == "a"
    assert report[DEV]["largest_group_share"] == pytest.approx(0.9)
    assert report[TRAIN]["labels"]["action"] == {"None": 95, "Spike": 5}
    assert any("dev: nhóm 'a' chiếm 90%" in w for w in warnings)
    assert any("dev: nhãn 'action' thiếu lớp ['Spike']" in w for w in warnings)


def test_split_report_is_quiet_when_balanced():
    groups = pd.Series(np.repeat(list("abcdefgh"), 10))
    labels = pd.Series(["None", "Spike"] * 40)
    index = {TRAIN: np.arange(40), DEV: np.arange(40, 80)}
    _, warnings = split_report(groups, index, {"action": labels})
    assert warnings == []


# ---------------------------------------------------------------- cross-validation
@pytest.mark.parametrize("mode,n_groups,expected_folds", [("kfold", 12, 5), ("lopo", 6, 6)])
def test_cv_folds_keep_groups_whole_and_validate_every_row_once(mode, n_groups, expected_folds):
    groups = pd.Series(np.repeat([f"g{i}" for i in range(n_groups)], 7))
    folds = cv_folds(groups, mode, n_folds=5)
    assert len(folds) == expected_folds

    seen = np.zeros(len(groups), dtype=int)
    for tr, va in folds:
        assert set(groups.iloc[tr]).isdisjoint(groups.iloc[va])
        seen[va] += 1
    assert (seen == 1).all()
    if mode == "lopo":
        assert all(groups.iloc[va].nunique() == 1 for _, va in folds)


def test_cv_folds_need_enough_groups():
    with pytest.raises(ValueError, match="kfold cần ít nhất 5"):
        cv_folds(pd.Series(["a", "b", "c"]), "kfold", n_folds=5)
    with pytest.raises(ValueError, match="lopo cần ít nhất 2"):
        cv_folds(pd.Series(["a", "a"]), "lopo")


def test_assert_no_overlap():
    assert_no_overlap(["m1", "m2"], ["m3"])
    with pytest.raises(ValueError, match="Rò rỉ"):
        assert_no_overlap(["m1", "m2"], ["m2", "m3"])
