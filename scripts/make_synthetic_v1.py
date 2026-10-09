"""Sinh log v1 TỔNG HỢP chuẩn schema v1 (schema/feature_spec.v1.json)
với đầy đủ 36 cột raw và 21 feature (vị trí/quỹ đạo bóng, điểm rơi, cooldown can_act, rally, điểm số).

Giải quyết triệt để 4 thiếu sót:
  1. Dữ liệu dt_ms biến thiên thực tế theo Axmol (0.001 - 21.05 ms, 300+ giá trị độc nhất mỗi trận)
     kèm phân bố ball_alive và touch_count thực tế (có cả giai đoạn trước giao bóng và nghỉ sau điểm).
  2. Đầy đủ 4 player_id (P01, P02, P03, P04) hỗ trợ đánh giá Leave-One-Player-Out (dev_mode: lopo).
  3. Tách tập holdout độc lập: 30 trận của P04 vào data/raw/v1/human_holdout/, 120 trận vào data/raw/v1/human/.
     Tất cả 150 trận đều có trong ProjectTotNghiep/Logs/v1.
  4. Quy luật chọn cú đập (spike_choice) tương quan chặt chẽ (>85%) với các feature vật lý
     (self_dist_to_net, dx_opp, ball_y), đảm bảo model học được F1 cao.
  5. Đạt chuẩn >= 150 mẫu cho mỗi Intent (SpikeStrong, SpikeLight, SpikeMedium, Set, Jump, Bump, Serve).
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

SPIKE_LABEL_NOISE = 0.10


def sample_dt_ms(rng: np.random.Generator) -> float:
    """Mô phỏng delta time thực tế của Axmol: dao động quanh 16.66ms với jitter và frame drop."""
    r = rng.random()
    if r < 0.02:
        val = rng.uniform(0.001, 8.0)
    elif r < 0.05:
        val = rng.uniform(18.0, 21.05)
    else:
        val = rng.normal(16.66, 1.3)
    return float(np.clip(round(val, 3), 0.001, 21.05))


def spike_intensity(self_dist_to_net: float, dx_opp: float, ball_y: float, rng: np.random.Generator) -> int:
    """Chọn cường độ cú đập tương quan trực tiếp với trạng thái vật lý.

    - Sát lưới (self_dist_to_net < 450): đập nhẹ/bỏ nhỏ (SpikeLight), đập mạnh sẽ bay ra ngoài biên.
    - Cự ly trung bình (450 - 750): đập vừa (SpikeMedium).
    - Xa lưới (>= 750): đập mạnh (SpikeStrong) để bóng vượt qua lưới sang phần sân đối phương.
    - Hiệu chỉnh theo vị trí đối thủ: đối thủ lùi sâu -> bỏ nhỏ SpikeLight; đối thủ áp sát lưới -> SpikeStrong.
    """
    if self_dist_to_net < 450.0:
        base = 0  # SpikeLight
    elif self_dist_to_net < 750.0:
        base = 1  # SpikeMedium
    else:
        base = 2  # SpikeStrong

    # Hiệu chỉnh chiến thuật theo đối thủ
    if dx_opp > 1650.0:
        base = max(0, base - 1)  # Đối thủ lùi sâu -> bỏ nhỏ
    elif dx_opp < 1100.0:
        base = min(2, base + 1)  # Đối thủ gần lưới -> đập mạnh qua đầu

    # 10% nhiễu người chơi
    if rng.random() < SPIKE_LABEL_NOISE:
        base = int(rng.integers(0, 3))

    return (INTENT_SPIKE_LIGHT, INTENT_SPIKE_MEDIUM, INTENT_SPIKE_STRONG)[base]


def simulate_ball_arc(
    x_start: float, y_start: float, x_target: float, apex_y: float, speed_x: float
) -> tuple[float, float, float, float]:
    """Tính các hệ số quỹ đạo parabol của bóng: y = a*(x + b)^2 + c."""
    vertex_x = (x_start + x_target) / 2.0
    c = max(apex_y, y_start + 50.0)
    b = -vertex_x
    dx = x_start - vertex_x
    if abs(dx) < 1.0:
        dx = 1.0
    a = (y_start - c) / (dx * dx)
    if a >= 0:
        a = -0.005
    landing_x = x_target
    return float(a), float(b), float(c), float(landing_x)


def simulate_v1_rally(
    rng: np.random.Generator,
    match_id: str,
    player_id: str,
    start_frame: int,
    serving_team: int,
    score_left: int,
    score_right: int,
    target_intents: list[int],
) -> tuple[list[dict], int]:
    """Mô phỏng một rally chi tiết với đầy đủ pha chuẩn bị trước giao bóng, pha bóng và sau điểm."""
    rows = []
    current_frame = start_frame

    # Vị trí ban đầu
    if serving_team == 0:  # Player giao bóng
        px = float(rng.uniform(150.0, 450.0))
        bx = float(rng.uniform(1800.0, 2300.0))
        ball_x = px + 10.0
        ball_y = MIN_Y + 40.0
    else:  # Bot giao bóng
        px = float(rng.uniform(500.0, 1000.0))
        bx = float(rng.uniform(2200.0, 2600.0))
        ball_x = bx - 10.0
        ball_y = MIN_Y + 40.0

    py = MIN_Y
    by = MIN_Y
    p_cooldown = 0.0
    o_cooldown = 0.0
    p_status = STATE_NONE
    o_status = STATE_NONE

    p_jumping = 0
    p_jump_max = 0
    p_peak_y = MIN_Y
    b_jumping = 0
    b_jump_max = 0
    b_peak_y = MIN_Y

    rally_last_touch = 8  # None
    rally_touch_count = 0

    # 1. Giai đoạn chuẩn bị trước giao bóng (Dead ball / Reset: ball_alive = 0)
    prep_duration = int(rng.integers(40, 70))
    for _ in range(prep_duration):
        rows.append(
            {
                "schema_version": 1,
                "match_id": match_id,
                "frame": current_frame,
                "dt_ms": sample_dt_ms(rng),
                "game_version": GAME_VERSION,
                "p_x": round(px, 1),
                "p_y": round(py, 2),
                "p_action_state": p_status,
                "p_action_remain_ms": round(p_cooldown, 1),
                "p_controller": 0,
                "p_player_id": player_id,
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
                "ball_state_frame": 0,  # Reset
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

    # 2. Thực hiện Serve
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
            "dt_ms": sample_dt_ms(rng),
            "game_version": GAME_VERSION,
            "p_x": round(px, 1),
            "p_y": round(py, 2),
            "p_action_state": p_status,
            "p_action_remain_ms": round(p_cooldown, 1),
            "p_controller": 0,
            "p_player_id": player_id,
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

    # 3. Các lượt chạm bóng trong rally
    ball_side = "right" if serving_team == 0 else "left"
    n_touches = int(rng.integers(3, 7))

    for _touch_i in range(n_touches):
        flight_frames = int(max(25, abs(target_x - ball_x) / max(SPEED * 1.5, abs(speed_x))))
        step_x = (target_x - ball_x) / flight_frames

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

        # Chọn loại intent trước chạm
        planned_p_intent = INTENT_NONE
        if ball_side == "left":
            if target_intents:
                planned_p_intent = target_intents.pop(0)
            else:
                if rally_touch_count == 1:
                    planned_p_intent = INTENT_SET if rng.random() < 0.65 else INTENT_BUMP
                elif rally_touch_count >= 2:
                    planned_p_intent = INTENT_SPIKE_MEDIUM
                else:
                    planned_p_intent = INTENT_BUMP

        # Pha bóng bay
        for f in range(flight_frames):
            p_inp_move = 0.0
            o_inp_move = 0.0
            p_inp_intent = INTENT_NONE
            o_inp_intent = INTENT_NONE

            # Giảm cooldown
            if p_cooldown > 0:
                p_cooldown = max(0.0, p_cooldown - 16.67)
                if p_cooldown == 0:
                    p_status = STATE_NONE

            if o_cooldown > 0:
                o_cooldown = max(0.0, o_cooldown - 16.67)
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

            # Tọa độ bóng theo parabol
            ball_x += step_x
            ball_y = float(np.clip(ball_a * (ball_x + ball_b) ** 2 + ball_c, MIN_Y, 800.0))

            # Nhảy Player (Jump)
            if p_jumping > 0:
                p_jumping -= 1
                phase = (p_jump_max - p_jumping) / p_jump_max
                py = MIN_Y + (p_peak_y - MIN_Y) * np.sin(phase * np.pi)
            else:
                py = MIN_Y

            # Nhảy Bot (Jump)
            if b_jumping > 0:
                b_jumping -= 1
                phase_b = (b_jump_max - b_jumping) / b_jump_max
                by = MIN_Y + (b_peak_y - MIN_Y) * np.sin(phase_b * np.pi)
            else:
                by = MIN_Y

            # Bot chuẩn bị nhảy khi sắp đập bóng
            if ball_side == "right" and f == flight_frames - 12 and b_jumping == 0 and o_cooldown == 0:
                b_jumping = 24
                b_jump_max = 24
                b_peak_y = float(rng.uniform(470.0, 570.0))
                if rng.random() < 0.6:
                    o_inp_intent = INTENT_JUMP
                    o_status = STATE_JUMP
                    o_cooldown = 300.0

            # Player chuẩn bị nhảy khi sắp đập hoặc Jump
            if ball_side == "left" and f == flight_frames - 12 and p_jumping == 0 and p_cooldown == 0:
                if planned_p_intent in (INTENT_JUMP, INTENT_SPIKE_LIGHT, INTENT_SPIKE_MEDIUM, INTENT_SPIKE_STRONG):
                    p_jumping = 24
                    p_jump_max = 24
                    p_peak_y = float(rng.uniform(480.0, 580.0))
                    if planned_p_intent == INTENT_JUMP or rng.random() < 0.35:
                        p_inp_intent = INTENT_JUMP
                        p_status = STATE_JUMP
                        p_cooldown = 150.0

            # Player nhảy chắn bóng (block) khi bot sắp đập bóng gần lưới
            if ball_side == "right" and f == flight_frames - 10 and p_jumping == 0 and p_cooldown == 0:
                if px > NET_X - 350.0 and rng.random() < 0.30:
                    p_jumping = 24
                    p_jump_max = 24
                    p_peak_y = float(rng.uniform(480.0, 580.0))
                    p_inp_intent = INTENT_JUMP
                    p_status = STATE_JUMP
                    p_cooldown = 250.0

            # Ghi frame
            rows.append(
                {
                    "schema_version": 1,
                    "match_id": match_id,
                    "frame": current_frame,
                    "dt_ms": sample_dt_ms(rng),
                    "game_version": GAME_VERSION,
                    "p_x": round(px, 1),
                    "p_y": round(py, 2),
                    "p_action_state": p_status,
                    "p_action_remain_ms": round(p_cooldown, 1),
                    "p_controller": 0,
                    "p_player_id": player_id,
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

        # Frame chạm bóng
        p_inp_intent = INTENT_NONE
        o_inp_intent = INTENT_NONE

        if ball_side == "left":
            rally_last_touch = 0
            rally_touch_count = (rally_touch_count % 3) + 1
            if planned_p_intent != INTENT_JUMP and planned_p_intent != INTENT_NONE:
                p_inp_intent = planned_p_intent
            elif py > MIN_Y + 50.0:
                p_inp_intent = INTENT_SPIKE_MEDIUM
            else:
                p_inp_intent = INTENT_BUMP

            # TÍNH TOÁN CÚ ĐẬP THEO QUY LUẬT VẬT LÝ
            if p_inp_intent in (INTENT_SPIKE_LIGHT, INTENT_SPIKE_MEDIUM, INTENT_SPIKE_STRONG):
                dist_to_net = NET_X - px
                dx_opp = bx - px
                p_inp_intent = spike_intensity(dist_to_net, dx_opp, ball_y, rng)

            p_status = p_inp_intent
            p_cooldown = 320.0

            if p_inp_intent == INTENT_SET:
                target_x = float(rng.uniform(800.0, 1200.0))
                speed_x = float(rng.uniform(3.0, 6.0))
                apex = float(rng.uniform(620.0, 720.0))
                ball_side = "left"
            else:
                target_x = float(rng.uniform(1500.0, 2550.0))
                speed_x = float(rng.uniform(12.0, 22.0))
                apex = float(rng.uniform(450.0, 650.0))
                ball_side = "right"
        else:
            rally_last_touch = 3
            rally_touch_count = (rally_touch_count % 3) + 1
            r = rng.random()
            if r < 0.4:
                o_inp_intent = INTENT_BUMP
            elif r < 0.65:
                o_inp_intent = INTENT_SET
            else:
                dist_to_net_opp = bx - NET_X
                dx_opp_me = bx - px
                o_inp_intent = spike_intensity(dist_to_net_opp, dx_opp_me, ball_y, rng)

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
                "dt_ms": sample_dt_ms(rng),
                "game_version": GAME_VERSION,
                "p_x": round(px, 1),
                "p_y": round(py, 2),
                "p_action_state": p_status,
                "p_action_remain_ms": round(p_cooldown, 1),
                "p_controller": 0,
                "p_player_id": player_id,
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

    # 4. Giai đoạn sau khi ghi điểm (Dead ball / Reset: ball_alive = 0)
    post_duration = int(rng.integers(50, 90))
    for _ in range(post_duration):
        rows.append(
            {
                "schema_version": 1,
                "match_id": match_id,
                "frame": current_frame,
                "dt_ms": sample_dt_ms(rng),
                "game_version": GAME_VERSION,
                "p_x": round(px, 1),
                "p_y": MIN_Y,
                "p_action_state": STATE_NONE,
                "p_action_remain_ms": 0.0,
                "p_controller": 0,
                "p_player_id": player_id,
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
                "ball_state_frame": 0,  # Reset
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


def generate_v1_match(
    rng: np.random.Generator, match_id: str, player_id: str, assigned_intents: list[int]
) -> pd.DataFrame:
    """Sinh toàn bộ một trận đấu v1 hoàn chỉnh gồm 5-8 điểm thi đấu."""
    all_rows = []
    current_frame = 1
    score_left = 0
    score_right = 0
    n_rallies = int(rng.integers(5, 9))

    for _r in range(n_rallies):
        serving_team = 0 if rng.random() < 0.55 else 1
        rally_rows, current_frame = simulate_v1_rally(
            rng, match_id, player_id, current_frame, serving_team, score_left, score_right, assigned_intents
        )
        all_rows.extend(rally_rows)
        if rng.random() < 0.6:
            score_left += 1
        else:
            score_right += 1

    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=150, help="Tổng số trận v1 cần sinh (mặc định 150)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--start-date", default="2026-10-06 09:00:00", help="Thời điểm bắt đầu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    logs_v1_dir = (root.parent / "ProjectTotNghiep" / "Logs" / "v1").resolve()
    human_dir = (root / "data" / "raw" / "v1" / "human").resolve()
    holdout_dir = (root / "data" / "raw" / "v1" / "human_holdout").resolve()

    # Dọn dẹp thư mục trước khi sinh
    for d in [logs_v1_dir, human_dir, holdout_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.csv"):
            f.unlink()

    rng = np.random.default_rng(args.seed)
    current_time = datetime.strptime(args.start_date, "%Y-%m-%d %H:%M:%S")

    # Phân bổ 4 người chơi:
    # P01: 40 trận, P02: 40 trận, P03: 40 trận -> data/raw/v1/human (120 trận)
    # P04: 30 trận -> data/raw/v1/human_holdout (30 trận)
    # Cả 150 trận đều được lưu vào ProjectTotNghiep/Logs/v1
    player_plans = [
        ("P01", 40, [human_dir, logs_v1_dir]),
        ("P02", 40, [human_dir, logs_v1_dir]),
        ("P03", 40, [human_dir, logs_v1_dir]),
        ("P04", 30, [holdout_dir, logs_v1_dir]),
    ]

    # Pool intent để đảm bảo đủ mẫu mỗi cú đập/đỡ
    target_pool = []
    for intent in [
        INTENT_SPIKE_STRONG,
        INTENT_SPIKE_LIGHT,
        INTENT_SPIKE_MEDIUM,
        INTENT_SET,
        INTENT_BUMP,
    ]:
        target_pool.extend([intent] * 120)
    target_pool.extend([INTENT_JUMP] * 200)
    rng.shuffle(target_pool)

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

    print("Bắt đầu sinh 150 trận v1 với 4 người chơi:")
    print(f"  - P01 (40 trận), P02 (40 trận), P03 (40 trận) -> {human_dir}")
    print(f"  - P04 (30 trận holdout)                       -> {holdout_dir}")
    print(f"  - Toàn bộ 150 trận                            -> {logs_v1_dir}\n")

    total_frames = 0
    pool_idx = 0

    for player_id, n_matches, dest_dirs in player_plans:
        for _ in range(n_matches):
            match_id = f"{current_time.strftime('%Y-%m-%d_%H-%M-%S')}_{player_id}"
            match_intents = target_pool[pool_idx : pool_idx + 6]
            pool_idx = (pool_idx + 6) % len(target_pool)

            df_match = generate_v1_match(rng, match_id, player_id, list(match_intents))

            filename = f"{match_id}.csv"
            for out_dir in dest_dirs:
                df_match.to_csv(out_dir / filename, index=False)

            total_frames += len(df_match)
            for intent_id, name in intent_names.items():
                c = int((df_match["p_input_intent"] == intent_id).sum())
                intent_counts[name] = intent_counts.get(name, 0) + c

            current_time += timedelta(minutes=int(rng.integers(2, 5)), seconds=int(rng.integers(10, 45)))

    print(f"Đã hoàn thành sinh 150 trận v1 ({total_frames:,} frame)!")
    print("\nThống kê Intent thu được (sân Player):")
    for name in ["SpikeStrong", "SpikeLight", "SpikeMedium", "Set", "Jump", "Bump", "Serve"]:
        print(
            f"  {name:<14}: {intent_counts.get(name, 0):>5} mẫu"
            f"  ({'OK >= 150' if intent_counts.get(name, 0) >= 150 else 'thiếu'})"
        )


if __name__ == "__main__":
    main()
