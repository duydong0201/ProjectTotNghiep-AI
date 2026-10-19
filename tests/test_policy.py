"""Test policy.py: Python ra quyết định y như AIInputSystem.cpp, và eval_policy chạy được đầu-cuối."""

import numpy as np
import pandas as pd
import pytest
import yaml

from spike_ai import eval_policy, policy
from spike_ai.schema import load_spec, raw_column_names

CLASSES = ["Bump", "Jump", "None", "Serve", "Set", "SpikeLight", "SpikeMedium", "SpikeStrong"]
K = {c: i for i, c in enumerate(CLASSES)}
SELF_X = 500.0


def _rows(n=1, **cols):
    """Dòng trạng thái tối thiểu của agent player. Bóng mặc định nằm ngay tâm nhân vật."""
    base = {
        "match_id": "m",
        "frame": np.arange(n),
        "p_x": SELF_X,
        "p_y": policy.MIN_Y,
        "ball_x": SELF_X + 14.6,  # tâm bóng trùng tâm nhân vật theo trục x
        "ball_y": policy.MIN_Y + 55.6,  # và theo trục y -> detect 0%
        "ball_speed": 0.0,
        "ball_a": 0.0,
        "ball_b": 0.0,
        "ball_c": 0.0,
        "ball_traj_type": policy.PARABOLIC,
        "ball_state_frame": policy.ALIVE,
        "serving_team": 0,
        "rally_last_touch": 8,
        "rally_last_touch_kind": 0,
    }
    base.update(cols)
    return pd.DataFrame({k: np.resize(v, n) if np.ndim(v) else [v] * n for k, v in base.items()})


def _proba(**p):
    out = np.zeros((1, len(CLASSES)))
    for c, v in p.items():
        out[0, K[c]] = v
    out[0, K["None"]] = 1.0 - sum(p.values())
    return out


def _dist(rows, proba, **params):
    return policy.action_distribution(rows, proba, CLASSES, "player", {**policy.GAME_POLICY, **params})


# ------------------------------------------------------------------ 1-2. ngưỡng và temperature
def test_below_threshold_the_bot_does_nothing():
    d = _dist(_rows(), _proba(SpikeMedium=0.25))
    assert d[0, K["None"]] == pytest.approx(1.0)


def test_above_threshold_the_bot_always_acts_even_if_none_is_the_argmax():
    """Đúng điểm khác argmax: P(None) = 0,65 vẫn ra tay vì P(ra tay) = 0,35 > 0,3."""
    d = _dist(_rows(), _proba(SpikeMedium=0.35))
    assert d[0, K["SpikeMedium"]] == pytest.approx(1.0)


def test_temperature_flattens_the_choice():
    d = _dist(_rows(), _proba(SpikeMedium=0.32, SpikeLight=0.04))
    ratio = (0.32 / 0.04) ** (1 / 3)
    assert d[0, K["SpikeMedium"]] / d[0, K["SpikeLight"]] == pytest.approx(ratio)
    assert d[0].sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ luật chặn
def test_model_never_proposes_jump_by_default():
    d = _dist(_rows(), _proba(Jump=0.5))
    assert d[0, K["Jump"]] == 0.0
    assert d[0, K["None"]] == pytest.approx(1.0)


def test_model_may_jump_when_the_switch_is_on():
    d = _dist(_rows(), _proba(Jump=0.5), model_may_jump=True)
    assert d[0, K["Jump"]] == pytest.approx(1.0)


def test_serve_only_on_my_serve_turn():
    alive = _dist(_rows(), _proba(Serve=0.5))
    assert alive[0, K["Serve"]] == 0.0
    my_turn = _dist(_rows(ball_state_frame=policy.RESET, serving_team=0), _proba(Serve=0.5))
    assert my_turn[0, K["Serve"]] == pytest.approx(1.0)
    their_turn = _dist(_rows(ball_state_frame=policy.RESET, serving_team=1), _proba(Serve=0.5))
    assert their_turn[0, K["Serve"]] == 0.0


def test_contact_out_of_reach_becomes_none_without_redrawing():
    """Bóng cách 80% chiều cao: Set (<= 100%) vẫn được, cú đập (< 50%) bị phủ quyết.

    Game phủ quyết SAU khi rút thăm, nên phần của cú đập thành None - không dồn sang Set.
    """
    rows = _rows(ball_y=policy.MIN_Y + 55.6 + 0.8 * 160)
    d = _dist(rows, _proba(Set=0.2, SpikeMedium=0.2))
    assert d[0, K["SpikeMedium"]] == 0.0
    assert d[0, K["Set"]] == pytest.approx(0.5)
    assert d[0, K["None"]] == pytest.approx(0.5)


def test_detect_thresholds_match_the_game():
    p = np.array([49.9, 50.0, 60.0, 60.1, 100.0, 100.1])
    assert policy.in_detect_zone(p, "SpikeLight").tolist() == [True, False, False, False, False, False]
    assert policy.in_detect_zone(p, "Bump").tolist() == [True, True, True, False, False, False]
    assert policy.in_detect_zone(p, "Set").tolist() == [True, True, True, True, True, False]


# ------------------------------------------------------------------ 4. luật nhảy sau Set
def _ball_at(height):
    """Bóng đứng yên ở độ cao `height`, thẳng trên đầu nhân vật (y = c, a = 0)."""
    return {"ball_c": height, "ball_y": height}


def test_air_contact_matches_the_jump_physics():
    # Nhân vật lên 265 -> 284,62 -> +9,81/frame; tâm cao hơn chân 64 px, bóng tâm +8,4.
    # Bóng ở 600: chạm khi y_t > 600 - 135,6 = 464,4 -> y_19 = 461,2 chưa, y_20 = 471,0 chạm.
    assert policy.frames_until_air_contact(_rows(**_ball_at(600.0)), "player", 70)[0] == 20
    # Quá cao, cả cú nhảy (đỉnh ~569) không với tới.
    assert policy.frames_until_air_contact(_rows(**_ball_at(2000.0)), "player", 70)[0] == -1


def test_jump_once_per_own_set_at_the_first_frame_in_the_window():
    rows = _rows(3, rally_last_touch=0, rally_last_touch_kind=policy.ACTION_STATE_SET, ball_c=[2000.0, 600.0, 600.0])
    proba = np.repeat(_proba(), 3, axis=0)
    d = _dist(rows, proba)
    assert d[:, K["Jump"]] == pytest.approx([0.0, 0.314, 0.0])


def test_no_jump_when_the_last_touch_was_not_my_set():
    rows = _rows(2, rally_last_touch=3, rally_last_touch_kind=policy.ACTION_STATE_SET, ball_c=600.0)
    d = _dist(rows, np.repeat(_proba(), 2, axis=0))
    assert (d[:, K["Jump"]] == 0).all()


def test_jump_gives_way_to_a_contact_chosen_by_the_model():
    """Game chỉ nhảy khi model không chọn cú chạm ở frame đó."""
    rows = _rows(1, rally_last_touch=0, rally_last_touch_kind=policy.ACTION_STATE_SET, ball_c=600.0)
    d = _dist(rows, _proba(Set=0.4))
    assert d[0, K["Set"]] == pytest.approx(1.0)
    assert d[0, K["Jump"]] == pytest.approx(0.0)


# ------------------------------------------------------------------ chỉ số kỳ vọng
def test_expected_confusion_and_scores():
    classes = ["None", "Set"]
    y = ["Set", "Set", "None", "None"]
    dist = np.array([[0.5, 0.5], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    cm = policy.expected_confusion(y, dist, classes)
    assert cm.tolist() == [[1.5, 0.5], [0.5, 1.5]]
    s = policy.scores_from_confusion(cm, classes, ["Set", "None"])
    assert s["per_class"]["Set"]["precision"] == pytest.approx(0.75)
    assert s["per_class"]["Set"]["recall"] == pytest.approx(0.75)
    assert s["macro_f1"] == pytest.approx(0.75)


def test_event_recall_counts_a_hit_one_frame_late():
    classes = ["None", "Set"]
    y = ["None", "Set", "None", "None"]
    dist = np.array([[1, 0], [1, 0], [0.5, 0.5], [1, 0]], dtype=float)
    ids = ["m"] * 4
    assert policy.event_recall(y, dist, classes, ids, tolerance=0)["Set"]["recall"] == pytest.approx(0.0)
    assert policy.event_recall(y, dist, classes, ids, tolerance=1)["Set"]["recall"] == pytest.approx(0.5)


def test_event_window_does_not_cross_matches():
    classes = ["None", "Set"]
    y = ["Set", "None"]
    dist = np.array([[1, 0], [0, 1]], dtype=float)
    assert policy.event_recall(y, dist, classes, ["a", "b"], tolerance=1)["Set"]["recall"] == pytest.approx(0.0)


def test_policy_overrides_are_checked():
    assert policy.policy_params({"policy": {"act_threshold": 0.5}})["act_threshold"] == 0.5
    with pytest.raises(ValueError, match="khong_co"):
        policy.policy_params({"policy": {"khong_co": 1}})


# ------------------------------------------------------------------ eval_policy đầu-cuối
SET_INTENT = load_spec("v1")["enums"]["FinalIntent"]["Set"]


def _write_match(folder, match_id, n=60):
    df = pd.DataFrame({c: np.zeros(n) for c in raw_column_names(load_spec("v1"))})
    df["schema_version"] = 1
    df["match_id"] = match_id
    df["frame"] = np.arange(n)
    df["p_player_id"], df["o_player_id"] = "P01", "BOT"
    df["p_controller"], df["o_controller"] = 0, 1
    df["rally_last_touch"] = 8
    df["ball_state_frame"] = -1
    df["p_x"], df["p_y"] = SELF_X, policy.MIN_Y
    df["ball_x"], df["ball_y"] = SELF_X + 14.6, policy.MIN_Y + 55.6
    df.loc[[10, 30, 50], "p_input_intent"] = SET_INTENT
    df.to_csv(folder / f"{match_id}.csv", index=False)


def test_eval_policy_end_to_end(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for i in range(5):
        _write_match(data, f"tran{i}")
    config = {
        "run_name": "pol",
        "schema": "v1",
        "data_dirs": [str(data)],
        "agent": "player",
        "targets": ["action"],
        "frames_around_action": 3,
        "label_shift": 1,
        "split": {"group_by": "match", "dev_mode": "kfold", "n_folds": 5, "salt": "spike-ai-split-v1"},
        "models": {"decision_tree": {"max_depth": 3}},
    }
    path = tmp_path / "pol.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    report = eval_policy.run(str(path), save=False)

    m = report["methods"]
    assert m["argmax / mọi frame"]["frames"] == m["bot trong game"]["frames"] == 5 * 59  # label_shift bỏ frame cuối
    assert m["argmax / cửa sổ"]["frames"] < m["argmax / mọi frame"]["frames"]
    assert report["actions_per_match"]["Set"]["nguoi_choi"] == pytest.approx(3.0)
