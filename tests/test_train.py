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


def _write_v0(folder: Path, names: list[str], bot_events: list[str] | None = None) -> None:
    rng = np.random.default_rng(0)
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        events = np.resize(bot_events, FRAMES) if bot_events else rng.choice(["None", "MoveLeft", "MoveRight"], FRAMES)
        pd.DataFrame(
            {
                "Frame": np.arange(FRAMES),
                "PlayerPosX": rng.uniform(100, 1300, FRAMES).round(),
                "PlayerPosY": 265,
                "PlayerEvent": "None",
                "BotPosX": rng.uniform(1450, 2600, FRAMES).round(),
                "BotPosY": 265,
                "BotEvent": events,
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
        label_shift=0,
    )

    metrics = train_module.run(config)

    assert metrics["lopo/move__spy"]["n_folds"] == 4
    # mỗi fold validate đúng 3 trận của một người
    assert _SpyModel.predicted_rows == [3 * FRAMES] * 4


# ---------------------------------------------------------------- nhãn có điều kiện (spike_choice)
# 10 frame / trận, trong đó đúng 3 frame là cú đập -> phần còn lại phải bị loại khỏi train và đo
SPIKE_CYCLE = ["SpikeLight", "None", "SpikeMedium", "MoveLeft", "SpikeStrong", "None", "Bump", "None", "None", "Serve"]
SPIKES_PER_MATCH = 3


def test_conditional_target_trains_only_on_rows_where_it_applies(env):
    tmp, write_config = env
    names = [f"match_{i:03d}" for i in range(30)]
    _write_v0(tmp / "data", names, bot_events=SPIKE_CYCLE)

    metrics = train_module.run(write_config(targets=["spike_choice"]))

    expected = _expected_frames(names)
    # fit trên số CÚ ĐẬP, không phải số frame
    assert _SpyModel.fitted_rows == [expected[TRAIN] // FRAMES * SPIKES_PER_MATCH]
    assert _SpyModel.predicted_rows == [expected[DEV] // FRAMES * SPIKES_PER_MATCH]
    assert set(metrics["dev/spike_choice__spy"]["support"]) <= {"SpikeLight", "SpikeMedium", "SpikeStrong"}


def test_conditional_target_keeps_other_targets_on_all_rows(env):
    """Trong cùng một run, `move` vẫn dùng mọi frame còn `spike_choice` chỉ dùng frame đập bóng."""
    tmp, write_config = env
    names = [f"match_{i:03d}" for i in range(30)]
    _write_v0(tmp / "data", names, bot_events=SPIKE_CYCLE)

    train_module.run(write_config(targets=["move", "spike_choice"]))

    n_train_matches = _expected_frames(names)[TRAIN] // FRAMES
    assert _SpyModel.fitted_rows == [n_train_matches * FRAMES, n_train_matches * SPIKES_PER_MATCH]


def test_conditional_target_without_any_sample_is_a_clear_error(env):
    tmp, write_config = env
    # không có cú đập nào trong dữ liệu
    _write_v0(tmp / "data", [f"match_{i:03d}" for i in range(20)], bot_events=["None", "MoveLeft"])

    with pytest.raises(ValueError, match="spike_choice.*không có mẫu nào"):
        train_module.run(write_config(targets=["spike_choice"]))


# ---------------------------------------------------------------- cửa sổ quanh lúc chạm bóng
def test_action_window_keeps_frames_around_each_action():
    df = pd.DataFrame({"match_id": ["m"] * 10, "frame": range(10)})
    y = pd.DataFrame({"action": ["None"] * 10})
    y.loc[5, "action"] = "Bump"

    keep = train_module.action_window_mask(df, y, half_width=2)

    assert list(np.flatnonzero(keep)) == [3, 4, 5, 6, 7]


def test_action_window_does_not_cross_match_boundary():
    """Hai trận nối nhau: frame cuối trận này không liên quan tới frame đầu trận sau."""
    df = pd.DataFrame({"match_id": ["m1"] * 5 + ["m2"] * 5, "frame": list(range(5)) * 2})
    y = pd.DataFrame({"action": ["None"] * 10})
    y.loc[4, "action"] = "Set"  # frame cuối của m1

    keep = train_module.action_window_mask(df, y, half_width=3)

    # chỉ lan trong m1, không chạm sang m2 (vị trí 5..9)
    assert list(np.flatnonzero(keep)) == [1, 2, 3, 4]


def test_action_window_keeps_every_action_sample():
    """Cắt frame None nhưng KHÔNG được mất một mẫu hành động nào."""
    rng = np.random.default_rng(0)
    n = 500
    acts = np.where(rng.random(n) < 0.02, "Bump", "None")
    df = pd.DataFrame({"match_id": ["m"] * n, "frame": range(n)})
    y = pd.DataFrame({"action": acts})

    keep = train_module.action_window_mask(df, y, half_width=5)

    assert keep[acts != "None"].all()
    assert keep.sum() < n  # thực sự có cắt


def test_action_window_on_match_without_any_action():
    df = pd.DataFrame({"match_id": ["m"] * 4, "frame": range(4)})
    y = pd.DataFrame({"action": ["None"] * 4})

    assert not train_module.action_window_mask(df, y, half_width=2).any()


# ------------------------------------------------- chi train tren frame nguoi dieu khien (DAgger)
HUMAN, RULE_BOT, ML_BOT = 0, 1, 2


def _dagger_log(n=12, match_id="m0", p_ctrl=HUMAN, o_ctrl=ML_BOT):
    """Log v1 toi thieu. p_ctrl / o_ctrl la so hoac danh sach theo tung frame."""
    df = pd.DataFrame({c: np.zeros(n) for c in raw_column_names(load_spec("v1"))})
    df["schema_version"] = 1
    df["match_id"] = match_id
    df["frame"] = np.arange(n)
    df["p_controller"] = p_ctrl
    df["o_controller"] = o_ctrl
    df["p_player_id"] = "P01"
    df["o_player_id"] = "BOT"
    return df


def test_only_human_drops_frames_played_by_the_rule_bot():
    """Tran DAgger: san trai la BOT LUAT. Train agent=player thi phai bo het."""
    df = pd.concat(
        [_dagger_log(match_id="nguoi", p_ctrl=HUMAN), _dagger_log(match_id="dagger", p_ctrl=RULE_BOT)],
        ignore_index=True,
    )
    rows, X, y = train_module.build_rows(df, "v1", "player", {})
    assert set(rows["match_id"]) == {"nguoi"}
    assert len(rows) == len(X) == len(y) == 12


def test_only_human_can_be_turned_off_on_purpose():
    df = _dagger_log(p_ctrl=RULE_BOT)
    rows, _, _ = train_module.build_rows(df, "v1", "player", {"only_human": False})
    assert len(rows) == 12


def test_dagger_keeps_only_the_frames_where_the_human_took_over():
    """San phai: bot ML lai, NGUOI chi gianh quyen o frame 4-7. Chi 4 frame do la nhan DAgger."""
    o_ctrl = [ML_BOT] * 4 + [HUMAN] * 4 + [ML_BOT] * 4
    df = _dagger_log(p_ctrl=RULE_BOT, o_ctrl=o_ctrl)
    rows, _, _ = train_module.build_rows(df, "v1", ["player", "opponent"], {})
    assert list(rows["_agent"]) == ["opponent"] * 4
    assert list(rows["frame"]) == [4, 5, 6, 7]


def test_dagger_merges_human_play_with_takeover_frames():
    """Du lieu cu (nguoi o san trai) + tran DAgger (nguoi gianh san phai) -> gop ca hai."""
    old = _dagger_log(match_id="cu", p_ctrl=HUMAN, o_ctrl=ML_BOT)
    dag = _dagger_log(match_id="dagger", p_ctrl=RULE_BOT, o_ctrl=[ML_BOT] * 9 + [HUMAN] * 3)
    rows, _, _ = train_module.build_rows(pd.concat([old, dag], ignore_index=True), "v1", ["player", "opponent"], {})
    counts = rows.groupby(["match_id", "_agent"]).size().to_dict()
    assert counts == {("cu", "player"): 12, ("dagger", "opponent"): 3}


def test_both_views_of_one_match_land_in_the_same_group():
    """Hai goc nhin cua CUNG mot tran phai cung nhom, neu khong se ro ri giua train va test."""
    from spike_ai.split import group_keys

    df = _dagger_log(p_ctrl=HUMAN, o_ctrl=[HUMAN] * 6 + [ML_BOT] * 6)
    rows, _, _ = train_module.build_rows(df, "v1", ["player", "opponent"], {})
    assert set(rows["_agent"]) == {"player", "opponent"}
    assert group_keys(rows, "match", ["player", "opponent"]).nunique() == 1


def test_group_by_player_reads_the_id_of_each_rows_own_agent():
    from spike_ai.split import group_keys

    df = _dagger_log(p_ctrl=HUMAN, o_ctrl=HUMAN)
    df["o_player_id"] = "P01_PHAI"
    rows, _, _ = train_module.build_rows(df, "v1", ["player", "opponent"], {})
    keys = group_keys(rows, "player", ["player", "opponent"])
    assert set(keys[rows["_agent"] == "player"]) == {"P01"}
    assert set(keys[rows["_agent"] == "opponent"]) == {"P01_PHAI"}


def test_nothing_left_after_filtering_is_a_clear_error():
    df = _dagger_log(p_ctrl=RULE_BOT)
    with pytest.raises(ValueError, match="Khong con frame nao"):
        train_module.build_rows(df, "v1", "player", {})


# ------------------------------------------------- trong so DAgger + loai tran
class _WeightSpy:
    """Model gia ghi lai sample_weight nhan duoc."""

    seen: list = []

    def fit(self, X, y, sample_weight=None):
        _WeightSpy.seen.append(None if sample_weight is None else np.asarray(sample_weight))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[0])

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)))
        out[:, 0] = 1.0
        return out


def test_agent_weight_is_attached_per_agent():
    df = _dagger_log(p_ctrl=HUMAN, o_ctrl=[HUMAN] * 4 + [ML_BOT] * 8)
    rows, _, _ = train_module.build_rows(df, "v1", ["player", "opponent"], {"agent_weight": {"opponent": 20}})
    assert set(rows.loc[rows["_agent"] == "player", "_weight"]) == {1.0}
    assert set(rows.loc[rows["_agent"] == "opponent", "_weight"]) == {20.0}


def test_fit_passes_sample_weight_only_when_given(monkeypatch):
    monkeypatch.setitem(model_zoo.REGISTRY, "wspy", lambda seed, **p: _WeightSpy())
    _WeightSpy.seen = []
    X, y = pd.DataFrame({"a": [0.0, 1.0, 2.0]}), pd.Series(["x", "y", "x"])
    train_module._fit("wspy", 0, {}, X, y, None)
    train_module._fit("wspy", 0, {}, X, y, np.array([1.0, 20.0, 1.0]))
    assert _WeightSpy.seen[0] is None
    assert list(_WeightSpy.seen[1]) == [1.0, 20.0, 1.0]


def test_fit_with_weights_works_through_a_pipeline():
    """logreg la Pipeline -> trong so phai di vao buoc cuoi, khong duoc bao loi."""
    X, y = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]}), pd.Series(["x", "y", "x", "y"])
    train_module._fit("logreg", 0, {}, X, y, np.array([1.0, 1.0, 5.0, 5.0]))


def test_fit_with_weights_on_unsupported_model_is_a_clear_error():
    """KNN khong nhan sample_weight. Am tham bo qua trong so thi so do sai ma khong ai biet."""
    X, y = pd.DataFrame({"a": [0.0, 1.0, 2.0]}), pd.Series(["x", "y", "x"])
    with pytest.raises(ValueError, match="khong ho tro sample_weight"):
        train_module._fit("knn", 0, {"n_neighbors": 1}, X, y, np.ones(3))


def test_run_applies_weights_and_exclude_matches(env, monkeypatch):
    tmp, write_config = env
    monkeypatch.setitem(model_zoo.REGISTRY, "wspy", lambda seed, **p: _WeightSpy())
    _WeightSpy.seen = []
    data = tmp / "data"
    data.mkdir(parents=True, exist_ok=True)
    for mid in ("that1", "that2", "mophong"):
        o_ctrl = [HUMAN] * 6 + [ML_BOT] * 6
        _dagger_log(match_id=mid, p_ctrl=RULE_BOT, o_ctrl=o_ctrl).to_csv(data / f"{mid}.csv", index=False)

    cfg = write_config(
        schema="v1",
        agent=["player", "opponent"],
        targets=["action"],
        split={
            "group_by": "match",
            "dev_mode": "kfold",
            "n_folds": 2,
            "dev_ratio": 0,
            "holdout_ratio": 0,
            "salt": SALT,
        },
        models={"wspy": {}},
        agent_weight={"opponent": 20},
        exclude_matches=["mophong"],
        label_shift=0,
    )
    train_module.run(cfg)
    # 2 fold CV + 1 lan fit cuoi. Moi lan fit chi thay frame nguoi o san phai, trong so 20.
    assert len(_WeightSpy.seen) == 3
    assert all(set(w) == {20.0} for w in _WeightSpy.seen)
    # lan fit cuoi tren ca pool: chi 2 tran that x 6 frame - tran mo phong da bi loai
    assert len(_WeightSpy.seen[-1]) == 12


# ------------------------------------------------- ghep nhan vao trang thai luc quyet dinh (ADR-011)
SET = 7  # FinalIntent.Set


def test_label_shift_pairs_each_label_with_the_frame_before_it():
    """Game ghi log SAU khi ap hanh dong: o dong co nhan Set, bong da mang quy dao Set.
    Nen nhan o frame k phai ghep voi feature o frame k-1."""
    df = _dagger_log(n=8, p_ctrl=HUMAN)
    df["p_input_intent"] = [0, 0, 0, 0, 0, SET, 0, 0]
    rows, _, y = train_module.build_rows(df, "v1", "player", {"label_shift": 1})
    got = dict(zip(rows["frame"], y["action"], strict=True))
    assert got[4] == "Set"  # trang thai frame 4 -> quyet dinh Set o frame 5
    assert got[5] == "None"
    assert 7 not in got  # frame cuoi khong con frame ke tiep -> bo
    assert len(rows) == 7


def test_label_shift_never_crosses_a_match_boundary():
    a = _dagger_log(n=4, match_id="a", p_ctrl=HUMAN)
    b = _dagger_log(n=4, match_id="b", p_ctrl=HUMAN)
    b["p_input_intent"] = [SET, 0, 0, 0]  # Set o frame DAU cua tran b
    rows, _, y = train_module.build_rows(pd.concat([a, b], ignore_index=True), "v1", "player", {"label_shift": 1})
    # dong cuoi cua tran a KHONG duoc nhan cu Set cua tran b
    assert "Set" not in set(y.loc[(rows["match_id"] == "a").to_numpy(), "action"])
    assert rows.groupby("match_id").size().to_dict() == {"a": 3, "b": 3}


def test_label_shift_uses_who_decided_not_who_was_in_control_before():
    """Lan gianh quyen DAgger bat dau dung frame 6: frame 5 (bot dieu khien) la trang thai,
    con quyet dinh o frame 6 la cua NGUOI -> phai giu."""
    o_ctrl = [ML_BOT] * 6 + [HUMAN] * 4
    df = _dagger_log(n=10, p_ctrl=RULE_BOT, o_ctrl=o_ctrl)
    rows, _, _ = train_module.build_rows(df, "v1", "opponent", {"label_shift": 1})
    assert list(rows["frame"]) == [5, 6, 7, 8]


def test_label_shift_keeps_move_labels_as_integers():
    df = _dagger_log(n=6, p_ctrl=HUMAN)
    df["p_input_move"] = [7.0, -7.0, 0.0, 7.0, 0.0, -7.0]
    _, _, y = train_module.build_rows(df, "v1", "player", {"label_shift": 1})
    assert y["move"].dtype.kind == "i"
    assert set(y["move"]) <= {-1, 0, 1}


def test_without_label_shift_nothing_changes():
    df = _dagger_log(n=8, p_ctrl=HUMAN)
    df["p_input_intent"] = [0, 0, 0, 0, 0, SET, 0, 0]
    rows, _, y = train_module.build_rows(df, "v1", "player", {})
    assert dict(zip(rows["frame"], y["action"], strict=True))[5] == "Set"
    assert len(rows) == 8


# ------------------------------------------------- label_shift bat buoc voi log v1/v2 (ADR-011)
def test_v1_config_without_label_shift_is_rejected(env):
    """Truoc day mac dinh la 0: chay lai config cu ra model ro ri nhan ma khong ai biet."""
    tmp, write_config = env
    _write_v1(tmp / "data", "m0", "P01", "v1")
    config = write_config(schema="v1", agent="player")
    with pytest.raises(ValueError, match="label_shift"):
        train_module.run(config)


def test_label_shift_zero_runs_but_warns(capsys):
    assert train_module.check_label_shift({"label_shift": 0}, "v1") == 0
    assert "RÒ RỈ NHÃN" in capsys.readouterr().out


def test_label_shift_one_is_silent(capsys):
    assert train_module.check_label_shift({"label_shift": 1}, "v2") == 1
    assert capsys.readouterr().out == ""


def test_v0_does_not_need_label_shift():
    assert train_module.check_label_shift({}, "v0") == 0


# ------------------------------------------------- tu hoc (self_imitation, ADR-014)
def _selfplay_log(match_id="auto0", o_pid="MLAUTO"):
    """24 frame, 2 pha bong: pha 1 (frame 0-7) bot ML (san phai) thang, pha 2 (12-19) bot luat thang."""
    df = _dagger_log(n=24, match_id=match_id, p_ctrl=RULE_BOT, o_ctrl=ML_BOT)
    df["o_player_id"] = o_pid
    df["ball_state_frame"] = 50
    df.loc[0:7, "ball_state_frame"] = -1
    df.loc[12:19, "ball_state_frame"] = -1
    df.loc[9:, "score_right"] = 1
    df.loc[21:, "score_left"] = 1
    return df


def test_won_rally_mask_marks_only_rallies_the_team_won():
    df = _selfplay_log()
    right = train_module.won_rally_mask(df, team=1)
    left = train_module.won_rally_mask(df, team=0)
    assert np.flatnonzero(right).tolist() == list(range(0, 8))
    assert np.flatnonzero(left).tolist() == list(range(12, 20))


def test_self_imitation_keeps_ml_frames_of_won_rallies_from_selfplay_matches_only():
    df = pd.concat([_selfplay_log("auto0"), _selfplay_log("nguoi_vs_bot", o_pid="BOT")], ignore_index=True)
    cfg = {"self_imitation": {"agent": "opponent", "player_id": "MLAUTO"}}
    rows, _, _ = train_module.build_rows(df, "v1", ["player", "opponent"], cfg)
    assert set(rows["_agent"]) == {"opponent"}  # san trai la bot luat -> bo
    assert set(rows["match_id"]) == {"auto0"}  # tran nguoi vs bot cu khong bi lan vao
    assert sorted(rows["frame"]) == list(range(0, 8))  # chi pha bot ML thang


def test_self_imitation_weight_is_separate_from_agent_weight():
    df = _selfplay_log()
    df.loc[22:23, "o_controller"] = HUMAN  # vai frame nguoi gianh quyen (DAgger)
    cfg = {
        "agent_weight": {"opponent": 20},
        "self_imitation": {"agent": "opponent", "player_id": "MLAUTO", "weight": 0.5},
    }
    rows, _, _ = train_module.build_rows(df, "v1", ["player", "opponent"], cfg)
    w = dict(zip(rows["frame"], rows["_weight"], strict=True))
    assert w[0] == 0.5 and w[22] == 20.0


def test_self_imitation_agent_must_be_trained():
    with pytest.raises(ValueError, match="self_imitation.agent"):
        train_module.build_rows(
            _selfplay_log(), "v1", "player", {"self_imitation": {"agent": "opponent", "player_id": "X"}}
        )


def _kill_log():
    """1 pha bong: bot luat giao (frame 2), bot ML Set (5) roi dap (9) va thang diem."""
    df = _dagger_log(n=20, match_id="auto_k", p_ctrl=RULE_BOT, o_ctrl=ML_BOT)
    df["o_player_id"] = "MLAUTO"
    df["rally_last_touch"] = 8
    df["ball_state_frame"] = 50
    df.loc[0:11, "ball_state_frame"] = -1
    df.loc[2:, ["rally_last_touch", "rally_touch_count"]] = [0, 1]
    df.loc[5:, ["rally_last_touch", "rally_touch_count"]] = [3, 1]
    df.loc[9:, ["rally_last_touch", "rally_touch_count"]] = [3, 2]
    df["rally_last_touch_kind"] = 0
    df.loc[2:, "rally_last_touch_kind"] = 4  # Serve
    df.loc[5:, "rally_last_touch_kind"] = 6  # Set
    df.loc[9:, "rally_last_touch_kind"] = 8  # SpikeMedium
    df.loc[13:, "score_right"] = 1
    return df


def test_kill_possession_keeps_only_the_scoring_possession():
    m = train_module.kill_possession_mask(_kill_log(), "opponent")
    assert np.flatnonzero(m).tolist() == list(range(3, 10))  # sau cu cham cuoi cua doi thu -> cu dap


def test_won_rally_without_own_spike_is_not_a_kill():
    df = _kill_log()
    df.loc[9:, "rally_last_touch_kind"] = 6  # cham cuoi la Set chu khong phai dap (diem do doi thu hong)
    assert not train_module.kill_possession_mask(df, "opponent").any()


def test_dev_mode_none_fits_on_everything_without_offline_metrics(env):
    tmp, write_config = env
    for i in range(4):
        _write_v1(tmp / "data", f"m{i}", "P01", "v1")
    config = write_config(
        schema="v1",
        agent="player",
        label_shift=0,
        split={"group_by": "match", "dev_mode": "none", "dev_ratio": 0, "holdout_ratio": 0, "salt": SALT},
    )
    metrics = train_module.run(config)
    assert metrics == {}
    assert _SpyModel.fitted_rows == [4 * FRAMES]


# ------------------------------------------------- hoc tu chuyen gia (train_on_controllers, ADR-015)
EXPERT = 3


def test_train_on_expert_frames_with_their_own_weight():
    df = _dagger_log(n=12, match_id="auto_e", p_ctrl=RULE_BOT, o_ctrl=[ML_BOT] * 4 + [EXPERT] * 6 + [HUMAN] * 2)
    cfg = {
        "train_on_controllers": ["human", "expert"],
        "agent_weight": {"opponent": 20},
        "controller_weight": {"expert": 2},
    }
    rows, _, _ = train_module.build_rows(df, "v1", ["player", "opponent"], cfg)
    w = dict(zip(rows["frame"], rows["_weight"], strict=True))
    assert sorted(w) == list(range(4, 12))  # bo frame bot ML va toan bo san trai (bot luat)
    assert w[4] == 2.0 and w[10] == 20.0


def test_default_still_learns_only_from_humans():
    df = _dagger_log(n=6, p_ctrl=RULE_BOT, o_ctrl=[EXPERT] * 3 + [HUMAN] * 3)
    rows, _, _ = train_module.build_rows(df, "v1", ["opponent"], {})
    assert sorted(rows["frame"]) == [3, 4, 5]


def test_unknown_teacher_is_rejected():
    with pytest.raises(ValueError, match="train_on_controllers"):
        train_module.build_rows(_dagger_log(), "v1", "player", {"train_on_controllers": ["than_thanh"]})


def test_expert_only_kills_keeps_only_scoring_expert_possessions():
    df = _kill_log()
    df["o_controller"] = EXPERT
    df.loc[18:19, "o_controller"] = HUMAN  # frame nguoi choi: khong bi loc
    cfg = {"train_on_controllers": ["human", "expert"], "expert_only_kills": True}
    rows, _, _ = train_module.build_rows(df, "v1", ["opponent"], cfg)
    assert sorted(rows["frame"]) == list(range(3, 10)) + [18, 19]


def test_expert_only_kills_drops_expert_possessions_that_did_not_score():
    df = _kill_log()
    df["o_controller"] = EXPERT
    df.loc[9:, "rally_last_touch_kind"] = 6  # diem do doi thu hong, khong phai cu dap cua chuyen gia
    cfg = {"train_on_controllers": ["human", "expert"], "expert_only_kills": True}
    with pytest.raises(ValueError, match="Khong con frame"):
        train_module.build_rows(df, "v1", ["opponent"], cfg)


def test_expert_kill_weight_upweights_scoring_possessions_and_keeps_the_rest():
    df = _kill_log()
    df["o_controller"] = EXPERT
    cfg = {"train_on_controllers": ["human", "expert"], "controller_weight": {"expert": 1}, "expert_kill_weight": 5}
    rows, _, _ = train_module.build_rows(df, "v1", ["opponent"], cfg)
    w = dict(zip(rows["frame"], rows["_weight"], strict=True))
    assert w[5] == 5.0  # trong luot ghi diem (frame 3-9)
    assert w[15] == 1.0  # ngoai luot ghi diem: van hoc, trong so thuong

