"""Test chặn rò rỉ dữ liệu giữa train / dev / holdout.

Các test này là "hàng rào": nếu ai đó (kể cả chính bạn vài tuần sau) sửa cách chia làm
cho holdout bị lộ vào train, hoặc đổi salt, test sẽ đỏ.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spike_ai import models as model_zoo
from spike_ai import train as train_module
from spike_ai.split import DEV, HOLDOUT, SPLITS, TRAIN, assign_split, group_keys, split_by_hash

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


# ---------------------------------------------------------------- train.py không dùng holdout
class _SpyModel:
    """Model giả ghi lại số dòng được fit và số dòng được predict."""

    fitted_rows: list[int] = []
    predicted_rows: list[int] = []

    def fit(self, X, y):
        _SpyModel.fitted_rows.append(len(X))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        _SpyModel.predicted_rows.append(len(X))
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)))
        out[:, 0] = 1.0
        return out


def _write_v0_matches(folder: Path, n_matches: int, frames: int = 10) -> None:
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True)
    for i in range(n_matches):
        pd.DataFrame(
            {
                "Frame": np.arange(frames),
                "PlayerPosX": rng.uniform(100, 1300, frames).round(),
                "PlayerPosY": 265,
                "PlayerEvent": "None",
                "BotPosX": rng.uniform(1450, 2600, frames).round(),
                "BotPosY": 265,
                "BotEvent": rng.choice(["None", "MoveLeft", "MoveRight"], frames),
            }
        ).to_csv(folder / f"match_{i:03d}.csv", index=False)


@pytest.mark.parametrize("final", [False, True])
def test_train_fits_only_on_train_split(tmp_path, monkeypatch, final):
    frames = 10
    _write_v0_matches(tmp_path / "data", n_matches=30, frames=frames)
    (tmp_path / "reports").mkdir()
    config = {
        "run_name": "spy",
        "schema": "v0",
        "data_dirs": [str(tmp_path / "data")],
        "agent": "opponent",
        "targets": ["move"],
        "split": {"group_by": "match", "dev_ratio": 0.2, "holdout_ratio": 0.2, "salt": SALT},
        "models": {"spy": {}},
    }
    config_path = tmp_path / "spy.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.setitem(model_zoo.REGISTRY, "spy", lambda seed, **p: _SpyModel())
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(train_module, "plot_confusion_matrix", lambda *a, **k: None)
    _SpyModel.fitted_rows, _SpyModel.predicted_rows = [], []

    metrics = train_module.run(str(config_path), final=final)

    expected = {name: 0 for name in SPLITS}
    for i in range(30):
        expected[assign_split(f"match_{i:03d}", SALT, 0.2, 0.2)] += frames

    assert _SpyModel.fitted_rows == [expected[TRAIN]]
    assert any(k.startswith("dev/") for k in metrics)
    assert any(k.startswith("holdout/") for k in metrics) == final
    # không có --final thì model chỉ predict trên dev, không bao giờ chạm holdout
    assert _SpyModel.predicted_rows == ([expected[DEV], expected[HOLDOUT]] if final else [expected[DEV]])
