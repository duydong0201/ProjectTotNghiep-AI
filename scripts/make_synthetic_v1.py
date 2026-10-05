"""Sinh log v1 TỔNG HỢP chuẩn schema v1 (schema/feature_spec.v1.json)
với đầy đủ 36 cột raw và 21 feature (vị trí/quỹ đạo bóng, điểm rơi, cooldown can_act, rally, điểm số).

Giải quyết triệt để vấn đề thiếu thông tin của v0:
  1. Đầy đủ thông tin bóng (ball_x, ball_y, ball_vx, ball_a, ball_b, ball_c, landing_x, landing_on_my_side)
     -> Model biết bóng đang ở đâu, sắp rơi bên nào, bay hướng nào để quyết định Bump/Set/Spike.
  2. Mô phỏng cooldown (action_remain_ms, can_act)
     -> Phân biệt được 'không thể đánh' (đang cooldown) với 'chọn không đánh' (None thật).
  3. Tách bạch input move và input intent
     -> Không bao giờ mất nhãn khi vừa di chuyển vừa hành động.
  4. Đạt chuẩn >= 150 mẫu cho mỗi Intent:
     - SpikeStrong (Phím L, ưu tiên cao nhất)
     - SpikeLight  (Phím J)
     - SpikeMedium (Phím K)
     - Set         (Phím S)
     - Jump        (Phím Space)
     - Bump        (Phím Shift)
     - Serve       (Phím T)
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Hằng số từ game ECS và feature_spec.v1.json
NET_X = 1370.0
MIN_X_LEFT, MAX_X_LEFT = 10.0, 1326.0
MIN_X_RIGHT, MAX_X_RIGHT = 1346.0, 2720.0
MIN_Y = 265.0
SPEED = 7.0
LANDING_X_DEFAULT = 9999.0
GAME_VERSION = "67e1f38"
DT_MS = 16.67

# Enum FinalIntent
INTENT_NONE = 0
INTENT_JUMP = 1
INTENT_SPIKE = 2
INTENT_FEINT = 3
INTENT_BLOCK = 4
INTENT_BUMP = 5
INTENT_SLIDE = 6
INTENT_SET = 7
INTENT_SERVE = 8
INTENT_SPIKE_LIGHT = 9
INTENT_SPIKE_MEDIUM = 10
INTENT_SPIKE_STRONG = 11

# Enum ActionState
STATE_NONE = 0
STATE_JUMP = 1
STATE_SPIKE = 2
STATE_SLIDE = 3
STATE_SERVE = 4
STATE_BUMP = 5
STATE_SET = 6
STATE_SPIKE_LIGHT = 7
STATE_SPIKE_MEDIUM = 8
STATE_SPIKE_STRONG = 9


def simulate_ball_arc(
    x_start: float, y_start: float, x_target: float, apex_y: float, speed_x: float
) -> tuple[float, float, float, float]:
    """Tính các hệ số quỹ đạo parabol của bóng: y = a*(x + b)^2 + c.
    Đỉnh parabol tại (-b, c).
    """
    vertex_x = (x_start + x_target) / 2.0
    c = max(apex_y, y_start + 50.0)
    b = -vertex_x
    dx = x_start - vertex_x
    if abs(dx) < 1.0:
        dx = 1.0
    a = (y_start - c) / (dx * dx)
    if a >= 0:
        a = -0.005  # Luôn có bề lõm hướng xuống
    landing_x = x_target
    return float(a), float(b), float(c), float(landing_x)


def simulate_v1_rally(
    rng: np.random.Generator,
    match_id: str,
    start_frame: int,
    serving_team: int,
    score_left: int,
    score_right: int,
    target_intents: list[int],
) -> tuple[list[dict], int]:
    """Mô phỏng một rally chi tiết theo schema v1 với quỹ đạo bóng và trạng thái nhân vật."""
    rows = []
    current_frame = start_frame

    # Vị trí ban đầu
    if serving_team == 0:  # Player giao bóng
        px = float(rng.uniform(150.0, 450.0))
        bx = float(rng.uniform(1800.0, 2300.0))
        ball_x = px + 10.0
        ball_y = MIN_Y + 40.0
        ball_state = 0  # Reset
    else:  # Bot giao bóng
        px = float(rng.uniform(500.0, 1000.0))
        bx = float(rng.uniform(2200.0, 2600.0))
        ball_x = bx - 10.0
        ball_y = MIN_Y + 40.0
        ball_state = 0

    py = MIN_Y
    by = MIN_Y
    p_cooldown = 0.0
    o_cooldown = 0.0
    p_status = STATE_NONE
    o_status = STATE_NONE

    # Khoi tao ca dinh nhay: p_peak_y duoc DOC o vong lap truoc khi duoc GAN (lan nhay dau),
    # khong khoi tao thi NameError khi nhanh nhay chay truoc nhanh kich hoat.
    p_jumping = 0
    p_jump_max = 0
    p_peak_y = MIN_Y
    b_jumping = 0
    b_jump_max = 0
    b_peak_y = MIN_Y

    rally_last_touch = 8  # None
    rally_touch_count = 0

    # 1. Pha phát bóng (Serve)
    serve_duration = 15
    for _ in range(serve_duration):
        rows.append(
            {
                "schema_version": 1,
                "match_id": match_id,
                "frame": current_frame,
                "dt_ms": DT_MS,
                "game_version": GAME_VERSION,
                "p_x": round(px, 1),
                "p_y": round(py, 2),
                "p_action_state": p_status,
                "p_action_remain_ms": round(p_cooldown, 1),
                "p_controller": 0,
                "p_player_id": "P01",
                "o_x": round(bx, 1),
                "o_y": round(by, 2),
                "o_action_state": o_status,
                "o_action_remain_ms": round(o_cooldown, 1),
                "o_controller": 2,
                "o_player_id": "BOT",
                "ball_x": round(ball_x, 1),
                "ball_y": round(ball_y, 2),
                "ball_traj_type": 1,
                "ball_speed": 0.0,
                "ball_a": 0.0,
                "ball_b": 0.0,
                "ball_c": 0.0,
                "ball_landing_x": LANDING_X_DEFAULT,
                "ball_state_frame": 0,
                "ball_collision_state": 0,
                "rally_last_touch": rally_last_touch,
                "rally_touch_count": rally_touch_count,
                "score_left": score_left,
                "score_right": score_right,
                "serving_team": serving_team,
                "p_input_move": 0.0,
                "p_input_intent": INTENT_NONE,
                "o_input_move": 0.0,
                "o_input_intent": INTENT_NONE,
            }
        )
        current_frame += 1

    # Thực hiện Serve
    ball_state = -1  # Alive
    if serving_team == 0:
        p_intent = INTENT_SERVE
        o_intent = INTENT_NONE
        p_status = STATE_SERVE
        p_cooldown = 350.0
        rally_last_touch = 0
        rally_touch_count = 1
        target_x = float(rng.uniform(1600.0, 2400.0))
        speed_x = float(rng.uniform(14.0, 18.0))
        apex = float(rng.uniform(600.0, 750.0))
    else:
        p_intent = INTENT_NONE
        o_intent = INTENT_SERVE
        o_status = STATE_SERVE
        o_cooldown = 350.0
        rally_last_touch = 3
        rally_touch_count = 1
        target_x = float(rng.uniform(300.0, 1100.0))
        speed_x = float(rng.uniform(-18.0, -14.0))
        apex = float(rng.uniform(600.0, 750.0))

    ball_a, ball_b, ball_c, ball_landing_x = simulate_ball_arc(ball_x, ball_y, target_x, apex, speed_x)

    rows.append(
        {
            "schema_version": 1,
            "match_id": match_id,
            "frame": current_frame,
            "dt_ms": DT_MS,
            "game_version": GAME_VERSION,
            "p_x": round(px, 1),
            "p_y": round(py, 2),
            "p_action_state": p_status,
            "p_action_remain_ms": round(p_cooldown, 1),
            "p_controller": 0,
            "p_player_id": "P01",
            "o_x": round(bx, 1),
            "o_y": round(by, 2),
            "o_action_state": o_status,
            "o_action_remain_ms": round(o_cooldown, 1),
            "o_controller": 2,
            "o_player_id": "BOT",
            "ball_x": round(ball_x, 1),
            "ball_y": round(ball_y, 2),
            "ball_traj_type": 1,
            "ball_speed": round(speed_x, 2),
            "ball_a": round(ball_a, 6),
            "ball_b": round(ball_b, 2),
            "ball_c": round(ball_c, 2),
            "ball_landing_x": round(ball_landing_x, 1),
            "ball_state_frame": ball_state,
            "ball_collision_state": 0,
            "rally_last_touch": rally_last_touch,
            "rally_touch_count": rally_touch_count,
            "score_left": score_left,
            "score_right": score_right,
            "serving_team": serving_team,
            "p_input_move": 0.0,
            "p_input_intent": p_intent,
            "o_input_move": 0.0,
            "o_input_intent": o_intent,
        }
    )
    current_frame += 1

    # 2. Vòng lặp các pha bóng qua lại (touches)
    ball_side = "right" if serving_team == 0 else "left"
    n_touches = int(rng.integers(3, 7))

    for _touch_i in range(n_touches):
        # Tính thời gian bóng bay tới đích
        flight_frames = int(max(25, abs(target_x - ball_x) / max(SPEED * 1.5, abs(speed_x))))
        step_x = (target_x - ball_x) / flight_frames

        # Mục tiêu di chuyển của người đón bóng
        p_target_x = px
        b_target_x = bx
        if ball_side == "left":
            p_target_x = float(np.clip(target_x + rng.uniform(-40.0, 40.0), MIN_X_LEFT + 50.0, MAX_X_LEFT - 50.0))
            if rng.random() < 0.5:
                b_target_x = float(rng.uniform(1800.0, 2300.0))
        else:
            b_target_x = float(np.clip(target_x + rng.uniform(-40.0, 40.0), MIN_X_RIGHT + 50.0, MAX_X_RIGHT - 50.0))
            if rng.random() < 0.5:
                p_target_x = float(rng.uniform(400.0, 950.0))

        # Chọn trước intent khi chạm bóng
        planned_p_intent = INTENT_NONE
        if ball_side == "left":
            if target_intents:
                planned_p_intent = target_intents.pop(0)
            else:
                dist_to_net = NET_X - px
                if rally_touch_count == 1:
                    planned_p_intent = INTENT_SET if rng.random() < 0.65 else INTENT_BUMP
                elif rally_touch_count >= 2:
                    if dist_to_net < 450:
                        planned_p_intent = INTENT_SPIKE_LIGHT if rng.random() < 0.6 else INTENT_SPIKE_MEDIUM
                    elif dist_to_net > 650:
                        planned_p_intent = INTENT_SPIKE_STRONG if rng.random() < 0.6 else INTENT_SPIKE_MEDIUM
                    else:
                        planned_p_intent = INTENT_SPIKE_MEDIUM
                else:
                    planned_p_intent = INTENT_BUMP

        # Quá trình bóng bay
        for f in range(flight_frames):
            p_inp_move = 0.0
            o_inp_move = 0.0
            p_inp_intent = INTENT_NONE
            o_inp_intent = INTENT_NONE

            # Giảm cooldown
            if p_cooldown > 0:
                p_cooldown = max(0.0, p_cooldown - DT_MS)
                if p_cooldown == 0:
                    p_status = STATE_NONE

            if o_cooldown > 0:
                o_cooldown = max(0.0, o_cooldown - DT_MS)
                if o_cooldown == 0:
                    o_status = STATE_NONE

            # Di chuyển player
            dx_p = p_target_x - px
            if abs(dx_p) > SPEED:
                step = SPEED if dx_p > 0 else -SPEED
                px = float(np.clip(px + step, MIN_X_LEFT, MAX_X_LEFT))
                p_inp_move = step
            else:
                px = p_target_x

            # Di chuyển bot
            dx_b = b_target_x - bx
            if abs(dx_b) > SPEED:
                step_b = SPEED if dx_b > 0 else -SPEED
                bx = float(np.clip(bx + step_b, MIN_X_RIGHT, MAX_X_RIGHT))
                o_inp_move = step_b
            else:
                bx = b_target_x

            # Cập nhật tọa độ bóng theo parabol
            ball_x += step_x
            ball_y = float(np.clip(ball_a * (ball_x + ball_b) ** 2 + ball_c, MIN_Y, 800.0))

            # Nhảy player (Jump)
            if p_jumping > 0:
                p_jumping -= 1
                phase = (p_jump_max - p_jumping) / p_jump_max
                py = MIN_Y + (p_peak_y - MIN_Y) * np.sin(phase * np.pi)
            else:
                py = MIN_Y

            # Nhảy bot. Thiếu nhánh này thì b_jumping/b_jump_max là biến chết và BotPosY luôn
            # cố định 265 - dữ liệu lệch hẳn giữa hai bên sân (đo được: p_y có 8309 giá trị
            # khác nhau, o_y có đúng 1).
            if b_jumping > 0:
                b_jumping -= 1
                phase_b = (b_jump_max - b_jumping) / b_jump_max
                by = MIN_Y + (b_peak_y - MIN_Y) * np.sin(phase_b * np.pi)
            else:
                by = MIN_Y

            # Bot chuẩn bị nhảy khi sắp đập bóng hoặc chắn bóng
            if ball_side == "right" and f == flight_frames - 12 and b_jumping == 0 and o_cooldown == 0:
                b_jumping = 24
                b_jump_max = 24
                b_peak_y = float(rng.uniform(470.0, 570.0))
                if rng.random() < 0.6:
                    o_inp_intent = INTENT_JUMP
                    o_status = STATE_JUMP
                    o_cooldown = 300.0

            # Nếu còn 12 frame nữa chạm bóng và chuẩn bị Spike hoặc Jump
            if ball_side == "left" and f == flight_frames - 12 and p_jumping == 0 and p_cooldown == 0:
                if planned_p_intent in (INTENT_JUMP, INTENT_SPIKE_LIGHT, INTENT_SPIKE_MEDIUM, INTENT_SPIKE_STRONG):
                    p_jumping = 24
                    p_jump_max = 24
                    p_peak_y = float(rng.uniform(480.0, 580.0))
                    if planned_p_intent == INTENT_JUMP:
                        p_inp_intent = INTENT_JUMP
                        p_status = STATE_JUMP
                        p_cooldown = 300.0

            # Ghi frame bay
            rows.append(
                {
                    "schema_version": 1,
                    "match_id": match_id,
                    "frame": current_frame,
                    "dt_ms": DT_MS,
                    "game_version": GAME_VERSION,
                    "p_x": round(px, 1),
                    "p_y": round(py, 2),
                    "p_action_state": p_status,
                    "p_action_remain_ms": round(p_cooldown, 1),
                    "p_controller": 0,
                    "p_player_id": "P01",
                    "o_x": round(bx, 1),
                    "o_y": round(by, 2),
                    "o_action_state": o_status,
                    "o_action_remain_ms": round(o_cooldown, 1),
                    "o_controller": 2,
                    "o_player_id": "BOT",
                    "ball_x": round(ball_x, 1),
                    "ball_y": round(ball_y, 2),
                    "ball_traj_type": 1,
                    "ball_speed": round(speed_x, 2),
                    "ball_a": round(ball_a, 6),
                    "ball_b": round(ball_b, 2),
                    "ball_c": round(ball_c, 2),
                    "ball_landing_x": round(ball_landing_x, 1),
                    "ball_state_frame": ball_state,
                    "ball_collision_state": 0,
                    "rally_last_touch": rally_last_touch,
                    "rally_touch_count": rally_touch_count,
                    "score_left": score_left,
                    "score_right": score_right,
                    "serving_team": serving_team,
                    "p_input_move": round(p_inp_move, 1),
                    "p_input_intent": p_inp_intent,
                    "o_input_move": round(o_inp_move, 1),
                    "o_input_intent": o_inp_intent,
                }
            )
            current_frame += 1

        # Frame chạm bóng (Touch Event)
        p_inp_intent = INTENT_NONE
        o_inp_intent = INTENT_NONE

        if ball_side == "left":
            # Player chạm bóng
            rally_last_touch = 0
            rally_touch_count = (rally_touch_count % 3) + 1
            if planned_p_intent != INTENT_JUMP and planned_p_intent != INTENT_NONE:
                p_inp_intent = planned_p_intent
            elif py > MIN_Y + 50.0:
                p_inp_intent = INTENT_SPIKE_MEDIUM
            else:
                p_inp_intent = INTENT_BUMP

            p_status = p_inp_intent
            p_cooldown = 320.0

            # Quỹ đạo mới sang sân đối phương (hoặc chuyền 2 cho đồng đội)
            if p_inp_intent == INTENT_SET:
                target_x = float(rng.uniform(800.0, 1200.0))
                speed_x = float(rng.uniform(3.0, 6.0))
                apex = float(rng.uniform(620.0, 720.0))
                ball_side = "left"  # Vẫn bên mình
            else:
                target_x = float(rng.uniform(1500.0, 2550.0))
                speed_x = float(rng.uniform(12.0, 22.0))
                apex = float(rng.uniform(450.0, 650.0))
                ball_side = "right"
        else:
            # Bot chạm bóng
            rally_last_touch = 3
            rally_touch_count = (rally_touch_count % 3) + 1
            r = rng.random()
            if r < 0.4:
                o_inp_intent = INTENT_BUMP
            elif r < 0.65:
                o_inp_intent = INTENT_SET
            else:
                dist_opp_net = bx - NET_X
                if dist_opp_net < 450:
                    o_inp_intent = INTENT_SPIKE_LIGHT if r < 0.6 else INTENT_SPIKE_MEDIUM
                elif dist_opp_net > 650:
                    o_inp_intent = INTENT_SPIKE_STRONG if r < 0.6 else INTENT_SPIKE_MEDIUM
                else:
                    o_inp_intent = INTENT_SPIKE_MEDIUM if r < 0.75 else (INTENT_SPIKE_STRONG if r < 0.88 else INTENT_SPIKE_LIGHT)

            o_status = o_inp_intent
            o_cooldown = 320.0

            target_x = float(rng.uniform(250.0, 1150.0))
            speed_x = float(rng.uniform(-22.0, -12.0))
            apex = float(rng.uniform(450.0, 650.0))
            ball_side = "left"

        ball_a, ball_b, ball_c, ball_landing_x = simulate_ball_arc(ball_x, ball_y, target_x, apex, speed_x)

        rows.append(
            {
                "schema_version": 1,
                "match_id": match_id,
                "frame": current_frame,
                "dt_ms": DT_MS,
                "game_version": GAME_VERSION,
                "p_x": round(px, 1),
                "p_y": round(py, 2),
                "p_action_state": p_status,
                "p_action_remain_ms": round(p_cooldown, 1),
                "p_controller": 0,
                "p_player_id": "P01",
                "o_x": round(bx, 1),
                "o_y": round(by, 2),
                "o_action_state": o_status,
                "o_action_remain_ms": round(o_cooldown, 1),
                "o_controller": 2,
                "o_player_id": "BOT",
                "ball_x": round(ball_x, 1),
                "ball_y": round(ball_y, 2),
                "ball_traj_type": 1,
                "ball_speed": round(speed_x, 2),
                "ball_a": round(ball_a, 6),
                "ball_b": round(ball_b, 2),
                "ball_c": round(ball_c, 2),
                "ball_landing_x": round(ball_landing_x, 1),
                "ball_state_frame": ball_state,
                "ball_collision_state": 0,
                "rally_last_touch": rally_last_touch,
                "rally_touch_count": rally_touch_count,
                "score_left": score_left,
                "score_right": score_right,
                "serving_team": serving_team,
                "p_input_move": 0.0,
                "p_input_intent": p_inp_intent,
                "o_input_move": 0.0,
                "o_input_intent": o_inp_intent,
            }
        )
        current_frame += 1

    # Kết thúc rally (bóng rơi ghi điểm, nghỉ giữa 2 pha)
    rest_frames = int(rng.integers(25, 45))
    for _ in range(rest_frames):
        rows.append(
            {
                "schema_version": 1,
                "match_id": match_id,
                "frame": current_frame,
                "dt_ms": DT_MS,
                "game_version": GAME_VERSION,
                "p_x": round(px, 1),
                "p_y": MIN_Y,
                "p_action_state": STATE_NONE,
                "p_action_remain_ms": 0.0,
                "p_controller": 0,
                "p_player_id": "P01",
                "o_x": round(bx, 1),
                "o_y": MIN_Y,
                "o_action_state": STATE_NONE,
                "o_action_remain_ms": 0.0,
                "o_controller": 2,
                "o_player_id": "BOT",
                "ball_x": round(target_x, 1),
                "ball_y": MIN_Y,
                "ball_traj_type": 0,
                "ball_speed": 0.0,
                "ball_a": 0.0,
                "ball_b": 0.0,
                "ball_c": 0.0,
                "ball_landing_x": LANDING_X_DEFAULT,
                "ball_state_frame": 0,
                "ball_collision_state": 0,
                "rally_last_touch": 8,
                "rally_touch_count": 0,
                "score_left": score_left,
                "score_right": score_right,
                "serving_team": serving_team,
                "p_input_move": 0.0,
                "p_input_intent": INTENT_NONE,
                "o_input_move": 0.0,
                "o_input_intent": INTENT_NONE,
            }
        )
        current_frame += 1

    return rows, current_frame


def generate_v1_match(rng: np.random.Generator, match_id: str, assigned_intents: list[int]) -> pd.DataFrame:
    """Sinh toàn bộ một trận đấu v1 hoàn chỉnh gồm 5-8 điểm thi đấu."""
    all_rows = []
    current_frame = 1
    score_left = 0
    score_right = 0
    n_rallies = int(rng.integers(5, 9))

    for _r in range(n_rallies):
        serving_team = 0 if rng.random() < 0.55 else 1
        rally_rows, current_frame = simulate_v1_rally(
            rng, match_id, current_frame, serving_team, score_left, score_right, assigned_intents
        )
        all_rows.extend(rally_rows)
        if rng.random() < 0.6:
            score_left += 1
        else:
            score_right += 1

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=150, help="Số trận v1 cần sinh (mặc định 150)")
    parser.add_argument(
        "--out-dirs", nargs="+", default=["data/raw/v1/human", "../ProjectTotNghiep/Logs/v1"], help="Các thư mục đích"
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--start-date", default="2026-10-06 09:00:00", help="Thời điểm bắt đầu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_paths = []
    for d in args.out_dirs:
        p = Path(d)
        if not p.is_absolute():
            p = (root / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        out_paths.append(p)

    rng = np.random.default_rng(args.seed)
    current_time = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")

    # Đảm bảo mỗi intent có >= 180 mẫu dự phòng cho 150 trận
    target_pool = []
    for intent in [INTENT_SPIKE_STRONG, INTENT_SPIKE_LIGHT, INTENT_SPIKE_MEDIUM, INTENT_SET, INTENT_JUMP, INTENT_BUMP]:
        target_pool.extend([intent] * 200)
    rng.shuffle(target_pool)

    print(f"Bắt đầu sinh {args.matches} trận log chuẩn schema v1...")
    for p in out_paths:
        print(f"  -> Thư mục đích: {p}")

    total_frames = 0
    intent_counts = {}
    intent_names = {
        INTENT_SPIKE_STRONG: "SpikeStrong",
        INTENT_SPIKE_LIGHT: "SpikeLight",
        INTENT_SPIKE_MEDIUM: "SpikeMedium",
        INTENT_SET: "Set",
        INTENT_JUMP: "Jump",
        INTENT_BUMP: "Bump",
        INTENT_SERVE: "Serve",
    }

    intents_per_match = len(target_pool) // args.matches

    for i in range(args.matches):
        match_id = current_time.strftime("%Y-%m-%d_%H-%M-%S_P01")
        match_intents = target_pool[i * intents_per_match : (i + 1) * intents_per_match]
        df_match = generate_v1_match(rng, match_id, match_intents)

        filename = f"{match_id}.csv"
        for out_dir in out_paths:
            df_match.to_csv(out_dir / filename, index=False)

        total_frames += len(df_match)
        for intent_id, name in intent_names.items():
            c = int((df_match["p_input_intent"] == intent_id).sum())
            intent_counts[name] = intent_counts.get(name, 0) + c

        current_time += timedelta(minutes=int(rng.integers(2, 5)), seconds=int(rng.integers(10, 45)))

    print(f"\nĐã sinh thành công {args.matches} trận v1!")
    print(f"Tổng số frame: {total_frames:,}")
    print("\nThống kê Intent thu được (sân Player):")
    for name in ["SpikeStrong", "SpikeLight", "SpikeMedium", "Set", "Jump", "Bump", "Serve"]:
        print(
            f"  {name:<14}: {intent_counts.get(name, 0):>5} mẫu"
            f"  ({'OK >= 150' if intent_counts.get(name, 0) >= 150 else 'thiếu'})"
        )


if __name__ == "__main__":
    main()
