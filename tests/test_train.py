"""Test train.py: model chỉ học trên đúng phần dữ liệu được phép, và holdout chỉ được đụng tới khi --final."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from spike_ai import models as model_zoo
from spike_ai import train as train_module
from spike_ai.schema import load_spec, raw_column_names
from spike_ai.split import DEV, HOLDOUT, SPLITS, TRAIN, assign_split

SALT = "spike-ai-split-v1"
FRAMES = 10


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


def _write_v0(folder: Path, names: list[str]) -> None:
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        pd.DataFrame(
            {
                "Frame": np.arange(FRAMES),
                "PlayerPosX": rng.uniform(100, 1300, FRAMES).round(),
                "PlayerPosY": 265,
                "PlayerEvent": "None",
                "BotPosX": rng.uniform(1450, 2600, FRAMES).round(),
                "BotPosY": 265,
                "BotEvent": rng.choice(["None", "MoveLeft", "MoveRight"], FRAMES),
            }
        ).to_csv(folder / f"{name}.csv", index=False)


def _write_v1(folder: Path, match_id: str, player: str, game_version: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({c: np.zeros(FRAMES) for c in raw_column_names(load_spec("v1"))})
    df["schema_version"] = 1
    df["match_id"] = match_id
    df["game_version"] = game_version
    df["p_player_id"], df["o_player_id"] = player, "BOT"
    df["p_input_move"] = np.resize([-7.0, 0.0, 7.0], FRAMES)
    df.to_csv(folder / f"{match_id}.csv", index=False)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Chạy train trong thư mục tạm, với model gián điệp."""
    monkeypatch.setitem(model_zoo.REGISTRY, "spy", lambda seed, **p: _SpyModel())
    monkeypatch.setattr(train_module, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(train_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(train_module, "plot_confusion_matrix", lambda *a, **k: None)
    (tmp_path / "reports").mkdir()
    _SpyModel.fitted_rows, _SpyModel.predicted_rows = [], []

    def write_config(**overrides) -> str:
        config = {
            "run_name": "spy",
            "schema": "v0",
            "data_dirs": [str(tmp_path / "data")],
            "agent": "opponent",
            "targets": ["move"],
            "split": {"group_by": "match", "dev_ratio": 0.2, "holdout_ratio": 0.2, "salt": SALT},
            "models": {"spy": {}},
        }
        config.update(overrides)
        path = tmp_path / "spy.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return str(path)

    return tmp_path, write_config


def _expected_frames(names, dev_ratio=0.2, holdout_ratio=0.2) -> dict:
    expected = {name: 0 for name in SPLITS}
    for n in names:
        expected[assign_split(n, SALT, dev_ratio, holdout_ratio)] += FRAMES
    return expected


# ---------------------------------------------------------------- dev_mode = single
@pytest.mark.parametrize("final", [False, True])
def test_single_fits_only_on_train_and_touches_holdout_only_when_final(env, final):
    tmp, write_config = env
    names = [f"match_{i:03d}" for i in range(30)]
    _write_v0(tmp / "data", names)

    metrics = train_module.run(write_config(), final=final)

    expected = _expected_frames(names)
    assert _SpyModel.fitted_rows == [expected[TRAIN]]
    assert any(k.startswith("dev/") for k in metrics)
    assert any(k.startswith("holdout/") for k in metrics) == final
    assert _SpyModel.predicted_rows == ([expected[DEV], expected[HOLDOUT]] if final else [expected[DEV]])


def test_final_without_any_holdout_is_an_error(env):
    tmp, write_config = env
    _write_v0(tmp / "data", [f"match_{i:03d}" for i in range(20)])
    config = write_config(split={"group_by": "match", "dev_ratio": 0.2, "holdout_ratio": 0, "salt": SALT})
    with pytest.raises(ValueError, match="không có holdout"):
        train_module.run(config, final=True)


# ---------------------------------------------------------------- dev_mode = kfold
def test_kfold_uses_all_non_holdout_data_and_never_holdout(env):
    tmp, write_config = env
    names = [f"match_{i:03d}" for i in range(30)]
    _write_v0(tmp / "data", names)
    split = {"group_by": "match", "dev_mode": "kfold", "n_folds": 5, "dev_ratio": 0, "holdout_ratio": 0.2, "salt": SALT}

    # cả nhãn số (move) lẫn nhãn chuỗi (action) - pandas 3 dùng StringDtype cho chuỗi
    metrics = train_module.run(write_config(split=split, targets=["move", "action"]))

    expected = _expected_frames(names, dev_ratio=0, holdout_ratio=0.2)
    pool = expected[TRAIN]
    for target in ("move", "action"):
        m = metrics[f"cv/{target}__spy"]
        assert m["n_folds"] == 5 and "macro_f1_std" in m
    # mỗi target: 5 lần fit trên (pool - 1 fold) + 1 lần fit model cuối trên toàn pool
    assert len(_SpyModel.fitted_rows) == 12
    assert _SpyModel.fitted_rows[5] == pool and _SpyModel.fitted_rows[11] == pool
    # mọi dòng của pool được validate đúng một lần, không dòng holdout nào được predict
    assert sum(_SpyModel.predicted_rows) == 2 * pool


# ---------------------------------------------------------------- holdout_dirs (holdout tương lai)
def test_holdout_dirs_are_loaded_only_when_final(env):
    tmp, write_config = env
    _write_v0(tmp / "data", [f"match_{i:03d}" for i in range(20)])
    _write_v0(tmp / "future", ["future_000", "future_001"])
    config = write_config(
        split={"group_by": "match", "dev_ratio": 0.2, "holdout_ratio": 0, "salt": SALT},
        holdout_dirs=[str(tmp / "future")],
    )

    assert not any(k.startswith("holdout/") for k in train_module.run(config))
    metrics = train_module.run(config, final=True)
    assert "holdout/move__spy" in metrics
    assert _SpyModel.predicted_rows[-1] == 2 * FRAMES


def test_holdout_dirs_overlapping_training_matches_is_rejected(env):
    tmp, write_config = env
    _write_v0(tmp / "data", [f"match_{i:03d}" for i in range(20)])
    _write_v0(tmp / "future", ["match_003"])  # trận này đã có trong data -> rò rỉ
    config = write_config(
        split={"group_by": "match", "dev_ratio": 0.2, "holdout_ratio": 0, "salt": SALT},
        holdout_dirs=[str(tmp / "future")],
    )
    with pytest.raises(ValueError, match="Rò rỉ"):
        train_module.run(config, final=True)


# ---------------------------------------------------------------- kiểm tra config
@pytest.mark.parametrize(
    "split,extra,message",
    [
        ({"dev_mode": "single", "dev_ratio": 0}, {}, "dev_ratio > 0"),
        ({"dev_mode": "kfold", "dev_ratio": 0.2}, {}, "dev_ratio: 0"),
        ({"dev_mode": "bogus", "dev_ratio": 0.2}, {}, "dev_mode"),
        ({"dev_ratio": 0.2, "holdout_ratio": 0.2}, {"holdout_dirs": ["x"]}, "Chỉ chọn một loại holdout"),
    ],
)
def test_invalid_split_config(split, extra, message):
    with pytest.raises(ValueError, match=message):
        train_module.validate_split_config({"split": {"salt": SALT, **split}, **extra})


# ---------------------------------------------------------------- v1: lọc phiên bản, LOPO
def test_game_versions_filter(tmp_path):
    _write_v1(tmp_path / "d", "m_old", "P01", "aaa1111")
    _write_v1(tmp_path / "d", "m_new", "P01", "bbb2222")
    df = train_module.load_data([tmp_path / "d"], "v1", ["bbb2222"])
    assert set(df["match_id"]) == {"m_new"}
    with pytest.raises(ValueError, match="Không còn dòng nào"):
        train_module.load_data([tmp_path / "d"], "v1", ["zzz"])


def test_lopo_leaves_out_one_player_per_fold(env):
    tmp, write_config = env
    players = ["P01", "P02", "P03", "P04"]
    for i in range(12):
        _write_v1(tmp / "data", f"m{i:02d}", players[i % 4], "v1")
    config = write_config(
        schema="v1",
        agent="player",
        split={"group_by": "match", "dev_mode": "lopo", "dev_ratio": 0, "holdout_ratio": 0, "salt": SALT},
    )

    metrics = train_module.run(config)

    assert metrics["lopo/move__spy"]["n_folds"] == 4
    # mỗi fold validate đúng 3 trận của một người
    assert _SpyModel.predicted_rows == [3 * FRAMES] * 4
