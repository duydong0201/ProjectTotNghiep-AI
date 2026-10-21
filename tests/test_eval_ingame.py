"""Test eval_ingame.py: các chỉ số trong game đếm đúng trên trận dựng tay."""

import numpy as np
import pandas as pd
import pytest

from spike_ai import eval_ingame
from spike_ai.schema import load_spec, raw_column_names

FI = load_spec("v1")["enums"]["FinalIntent"]
AS = load_spec("v1")["enums"]["ActionState"]
PLAYER, OPP, NONE = 0, 3, 8
P_X, GROUND = 500.0, 265.0
FAR = {"ball_x": 1370.0, "ball_y": 1200.0}  # trên lưới, xa cả hai nhân vật
NEAR_PLAYER = {"ball_x": P_X + 14.6, "ball_y": GROUND + 55.6}  # tâm bóng trùng tâm người chơi


def _match(n=40, v2=True):
    cols = raw_column_names(load_spec("v1"))
    df = pd.DataFrame({c: np.zeros(n) for c in cols})
    df["schema_version"] = 1
    df["match_id"] = "m"
    df["frame"] = np.arange(n)
    df["p_x"], df["p_y"] = P_X, GROUND
    df["o_x"], df["o_y"] = 2000.0, GROUND
    df["ball_x"], df["ball_y"] = FAR["ball_x"], FAR["ball_y"]
    df["ball_state_frame"] = -1
    df["rally_last_touch"] = NONE
    df["p_controller"], df["o_controller"] = 0, 2
    df["p_input_move"] = 7.0  # đang chạy, để không tính là đứng yên
    if v2:
        df["rally_last_touch_kind"] = 0
    return df


def _touch(df, row, entity, kind, intent=None, prefix="p_"):
    """Cú chạm game chấp nhận từ `row`: người chạm cuối + số chạm đổi, cột intent ghi phím bấm."""
    df.loc[row:, "rally_last_touch"] = entity
    df.loc[row:, "rally_touch_count"] = df.loc[row - 1, "rally_touch_count"] + 1
    if "rally_last_touch_kind" in df:
        df.loc[row:, "rally_last_touch_kind"] = AS[kind]
    df.loc[row, f"{prefix}input_intent"] = FI[intent or kind]


def test_points_and_win_rate():
    df = _match()
    df.loc[10:, "score_left"] = 1
    df.loc[20:, "score_left"] = 2
    df.loc[30:, "score_right"] = 1
    left = eval_ingame.side_metrics(df, "trái")
    right = eval_ingame.side_metrics(df, "phải")
    assert (left["points_won"], left["points_lost"]) == (2, 1)
    assert (right["points_won"], right["points_lost"]) == (1, 2)
    assert eval_ingame.summarize(left, 1)["pct_diem_thang"] == pytest.approx(200 / 3)


def test_spike_attempt_is_not_a_hit_unless_the_game_accepted_it():
    df = _match()
    df.loc[5, "p_input_intent"] = FI["SpikeMedium"]  # bấm, nhưng game bỏ qua
    _touch(df, 15, PLAYER, "SpikeMedium")  # bấm và trúng
    s = eval_ingame.summarize(eval_ingame.side_metrics(df, "trái"), 1)
    assert s["cu_dap_bam_moi_tran"] == 2
    assert s["cu_dap_trung_moi_tran"] == 1


def test_holding_a_key_counts_one_action():
    df = _match()
    df.loc[5:9, "p_input_intent"] = FI["Bump"]
    assert eval_ingame.side_metrics(df, "trái")["actions"] == 1


def test_set_jump_spike_chain_and_random_jump():
    df = _match()
    df.loc[3, "p_input_intent"] = FI["Jump"]  # nhảy khi chưa có cú Set nào -> bừa
    df.loc[4:8, "p_y"] = 300.0
    _touch(df, 10, PLAYER, "Set")
    df.loc[15, "p_input_intent"] = FI["Jump"]  # nhảy ngay sau cú Set của mình -> hợp lệ
    df.loc[16:25, "p_y"] = 400.0
    _touch(df, 22, PLAYER, "SpikeStrong")
    m = eval_ingame.side_metrics(df, "trái")
    assert m["jumps"] == 2
    assert m["random_jumps"] == 1
    assert m["set_jump_spike"] == 1


def test_jump_on_my_serve_turn_is_not_random():
    df = _match()
    df.loc[0:9, "ball_state_frame"] = 0
    df["serving_team"] = 0
    df.loc[5, "p_input_intent"] = FI["Jump"]
    assert eval_ingame.side_metrics(df, "trái")["random_jumps"] == 0


def test_idle_counts_only_live_ball_frames_without_move_or_action():
    df = _match(n=10)
    df.loc[0:3, "p_input_move"] = 0.0  # đứng yên 4 frame
    df.loc[3, "p_input_intent"] = FI["Set"]  # ... nhưng frame 3 có làm gì đó
    df.loc[8:9, "ball_state_frame"] = 0  # bóng chết: không tính
    df.loc[8:9, "p_input_move"] = 0.0
    m = eval_ingame.side_metrics(df, "trái")
    assert (m["idle_frames"], m["alive_frames"]) == (3, 8)


def test_ball_in_reach_without_a_touch_is_a_miss():
    df = _match()
    for k, v in NEAR_PLAYER.items():
        df.loc[5:7, k] = v  # lần 1: lọt vào tầm, không chạm
        df.loc[20:22, k] = v  # lần 2: lọt vào tầm và chạm
    _touch(df, 21, PLAYER, "Bump")
    m = eval_ingame.side_metrics(df, "trái")
    assert (m["zone_episodes"], m["zone_missed"]) == (2, 1)


def test_loss_cause_names_the_last_touch():
    df = _match()
    _touch(df, 5, OPP, "SpikeMedium", prefix="o_")
    _touch(df, 10, PLAYER, "Bump")
    df.loc[20:, "ball_state_frame"] = 50
    df.loc[20:, "score_right"] = 1
    assert eval_ingame.side_metrics(df, "trái")["loss_causes"] == {"mình chạm cuối bằng Bump": 1}


def test_v1_log_without_kind_column_rebuilds_it():
    df = _match(v2=False)
    _touch(df, 10, PLAYER, "Set")
    assert eval_ingame.side_metrics(df, "trái")["hits"] == {"Set": 1}


def test_controller_names():
    df = _match()
    assert eval_ingame.controller_of(df, "p_") == "người"
    df.loc[:9, "o_controller"] = 0
    assert eval_ingame.controller_of(df, "o_") == "bot ML + người"


def test_evaluate_reads_a_folder_and_skips_feature_dumps(tmp_path):
    for name in ("a", "b"):
        df = _match(v2=False)
        df["match_id"] = name
        df.loc[20:, "score_left"] = 1
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "a_feat.csv").write_text("khong,phai,log\n1,2,3\n", encoding="utf-8")
    report = eval_ingame.evaluate([tmp_path])
    left = report["sides"]["trái"]
    assert left["tran"] == 2
    assert left["diem_thang"] == 2
    assert left["ai"] == "người"
    assert report["sides"]["phải"]["ai"] == "bot ML"
