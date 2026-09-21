"""Biến log thô (raw) thành ma trận feature X và nhãn y.

QUY TẮC VÀNG
------------
Mọi phép biến đổi trong file này phải viết lại được Y HỆT trong C++ (MLInputSystem),
vì lúc chạy trong game, C++ sẽ tự tính feature từ ComponentStorage rồi đưa vào model.
Chỉ dùng phép toán đơn giản: cộng/trừ, so sánh, đổi dấu. Thứ tự cột của X lấy từ
schema/*.json - không tự sắp xếp lại.

HỆ TOẠ ĐỘ "GÓC NHÌN CỦA MÌNH" (agent view)
-----------------------------------------
Sân bên phải được lật qua lưới: x' = 2 * NET_X - x. Nhờ vậy dù học nhân vật bên trái
(player) hay bên phải (opponent), model luôn thấy "mình ở bên trái, đánh sang phải".
Nhãn move cũng được lật theo: +1 luôn có nghĩa là tiến về phía lưới.
Khi đưa kết quả về game, nhân vật bên phải phải lật ngược lại: moveX = -move.
"""

import numpy as np
import pandas as pd

from .schema import feature_names, load_spec

AGENT_SIDE = {"player": "left", "opponent": "right"}

V0_ACTIONS = {"Jump", "Spike", "Slide", "Serve", "Bump", "Set", "SpikeLight", "SpikeMedium", "SpikeStrong"}


def to_view_x(x, side: str, net_x: float):
    """Chuyển toạ độ x của game sang hệ toạ độ góc nhìn của agent."""
    return x if side == "left" else 2.0 * net_x - x


def to_view_sign(v, side: str):
    """Đổi dấu đại lượng có hướng (vận tốc, move) theo phía sân."""
    return v if side == "left" else -v


def build(df: pd.DataFrame, version: str, agent: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (X, y). X có đúng các cột feature theo spec; y có cột 'move' và 'action'."""
    if agent not in AGENT_SIDE:
        raise ValueError(f"agent phải là {list(AGENT_SIDE)}, nhận: {agent}")
    spec = load_spec(version)
    builder = {"v0": _build_v0, "v1": _build_v1}[version]
    X, y = builder(df, agent, spec)

    expected = feature_names(spec)
    if list(X.columns) != expected:
        raise AssertionError(f"Thứ tự feature khác spec.\ncode: {list(X.columns)}\nspec: {expected}")
    return X.astype("float32"), y


# ---------------------------------------------------------------------------
# v0: format CSV cũ (chỉ có vị trí 2 nhân vật + event)
# ---------------------------------------------------------------------------
def _build_v0(df: pd.DataFrame, agent: str, spec: dict):
    net_x = spec["constants"]["NET_X"]
    side = AGENT_SIDE[agent]
    me, opp = ("Player", "Bot") if agent == "player" else ("Bot", "Player")

    self_x = to_view_x(df[f"{me}PosX"].astype(float), side, net_x)
    opp_x = to_view_x(df[f"{opp}PosX"].astype(float), side, net_x)

    X = pd.DataFrame(
        {
            "self_x": self_x,
            "self_y": df[f"{me}PosY"].astype(float),
            "opp_x": opp_x,
            "opp_y": df[f"{opp}PosY"].astype(float),
            "dx_opp": opp_x - self_x,
            "self_dist_to_net": net_x - self_x,
        }
    )

    event = df[f"{me}Event"].astype(str)
    move = event.map({"MoveLeft": -1, "MoveRight": 1}).fillna(0).astype(int)
    y = pd.DataFrame(
        {
            "move": to_view_sign(move, side),
            "action": event.where(event.isin(V0_ACTIONS), "None"),
        }
    )
    return X, y


# ---------------------------------------------------------------------------
# v1: format mới (có bóng, rally, tỷ số, input) - xem schema/feature_spec.v1.json
# ---------------------------------------------------------------------------
def _build_v1(df: pd.DataFrame, agent: str, spec: dict):
    c = spec["constants"]
    net_x = c["NET_X"]
    side = AGENT_SIDE[agent]
    me, opp = ("p_", "o_") if agent == "player" else ("o_", "p_")
    my_entity = c["PLAYER_ENTITY"] if agent == "player" else c["OPPONENT_ENTITY"]
    opp_entity = c["OPPONENT_ENTITY"] if agent == "player" else c["PLAYER_ENTITY"]
    team = spec["enums"]["Team"]
    my_team = team["LEFT"] if side == "left" else team["RIGHT"]

    def vx(col):
        return to_view_x(df[col].astype(float), side, net_x)

    self_x = vx(f"{me}x")
    ball_x = vx("ball_x")

    # parabol y = a*(x+b)^2 + c có đỉnh tại x = -b
    a = df["ball_a"].astype(float)
    vertex_x = to_view_x(-df["ball_b"].astype(float), side, net_x)
    vertex_dx = np.where(a != 0.0, vertex_x - self_x, 0.0)

    landing_raw = df["ball_landing_x"].astype(float)
    landing_valid = landing_raw < c["LANDING_X_VALID_BELOW"]
    landing_x = to_view_x(landing_raw, side, net_x)

    my_score = df["score_left"] if side == "left" else df["score_right"]
    opp_score = df["score_right"] if side == "left" else df["score_left"]

    X = pd.DataFrame(
        {
            "self_x": self_x,
            "self_y": df[f"{me}y"].astype(float),
            "opp_x": vx(f"{opp}x"),
            "self_dist_to_net": net_x - self_x,
            "ball_dx": ball_x - self_x,
            "ball_y": df["ball_y"].astype(float),
            "ball_vx": to_view_sign(df["ball_speed"].astype(float), side),
            "ball_a": a,
            "ball_c": df["ball_c"].astype(float),
            "ball_vertex_dx": vertex_dx,
            "landing_valid": landing_valid.astype(float),
            "landing_dx": np.where(landing_valid, landing_x - self_x, 0.0),
            "landing_on_my_side": (landing_valid & (landing_x < net_x)).astype(float),
            "ball_alive": (df["ball_state_frame"] == -1).astype(float),
            "touch_count": df["rally_touch_count"].astype(float),
            "last_touch_me": (df["rally_last_touch"] == my_entity).astype(float),
            "last_touch_opp": (df["rally_last_touch"] == opp_entity).astype(float),
            "can_act": (df[f"{me}action_state"] == 0).astype(float),
            "action_remain_ms": df[f"{me}action_remain_ms"].astype(float),
            "my_serve": (df["serving_team"] == my_team).astype(float),
            "score_diff": (my_score - opp_score).astype(float),
        }
    )

    intent_names = {v: k for k, v in spec["enums"]["FinalIntent"].items()}
    move = np.sign(df[f"{me}input_move"].astype(float)).astype(int)
    y = pd.DataFrame(
        {
            "move": to_view_sign(move, side),
            "action": df[f"{me}input_intent"].map(intent_names).fillna("None"),
        }
    )
    return X, y
