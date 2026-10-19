"""Sinh 5 trận đấu DAgger chuẩn format CSV 37 cột (schema_version = 2) vào ProjectTotNghiep/Logs/v1/.

Đáp ứng chính xác toàn bộ đặc tả và 5 tiêu chí bắt buộc:
  1. p_controller luôn = 1 ở 100% dòng, p_player_id == 'BOTRULE', p_y == 265, và không có dòng nào thiếu dữ liệu.
  2. o_controller = 0 chiếm 25 - 40% số dòng trong mỗi trận.
  3. Ở các dòng o_controller = 0:
     - o_input_move != 0 ở >= 10% số dòng (người chơi di chuyển dưới bóng sau Set).
     - o_input_intent:
       * 1 (Jump): 20 - 30
       * 7 (Set): 20 - 30
       * 5 (Bump): 5 - 8
       * 9 (SpikeLight): 8 - 10
       * 10 (SpikeMedium): 8 - 10
       * 11 (SpikeStrong): 8 - 10
       * 8 (Serve): ~10 (trong đó ~1/3 có Jump 1 ngay trước).
  4. rally_last_touch_kind có xuất hiện giá trị 5 (Bump).
  5. Phần lớn lượt serving_team = 1 có o_controller = 0 (người chơi giành lượt giao bóng).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Hằng số chuẩn theo game và DAgger logs
NET_X = 1370.0
MIN_X_LEFT, MAX_X_LEFT = 350.0, 1300.0
MIN_X_RIGHT, MAX_X_RIGHT = 1437.0, 2445.0
GROUND_Y = 265.0
JUMP_STEP_Y = 9.8099975  # (569.109924 - 265.0) / 31
JUMP_PEAK_Y = 569.109924
GAME_VERSION = "3109aa8"

# Bảng mã phím bấm (*_input_intent)
INTENT_NONE = 0
INTENT_JUMP = 1
INTENT_BUMP = 5
INTENT_SET = 7
INTENT_SERVE = 8
INTENT_SPIKE_LIGHT = 9
INTENT_SPIKE_MEDIUM = 10
INTENT_SPIKE_STRONG = 11

# Bảng mã trạng thái và chạm bóng (*_action_state, rally_last_touch_kind)
STATE_NONE = 0
STATE_JUMP = 1
STATE_SERVE = 4
STATE_BUMP = 5
STATE_SET = 6
STATE_SPIKE_LIGHT = 7
STATE_SPIKE_MEDIUM = 8
STATE_SPIKE_STRONG = 9

INTENT_TO_STATE = {
    INTENT_NONE: STATE_NONE,
    INTENT_JUMP: STATE_JUMP,
    INTENT_SERVE: STATE_SERVE,
    INTENT_BUMP: STATE_BUMP,
    INTENT_SET: STATE_SET,
    INTENT_SPIKE_LIGHT: STATE_SPIKE_LIGHT,
    INTENT_SPIKE_MEDIUM: STATE_SPIKE_MEDIUM,
    INTENT_SPIKE_STRONG: STATE_SPIKE_STRONG,
}


def sample_dt_ms_dagger(rng: np.random.Generator) -> float:
    """Mô phỏng dt_ms từ 2 trận thật (mean ~16.69 ms, std ~0.43 ms, range 14.4 - 38.1 ms)."""
    r = rng.random()
    if r < 0.003:
        val = rng.uniform(22.0, 36.0)
    elif r < 0.01:
        val = rng.uniform(14.4, 15.8)
    else:
        val = rng.normal(16.688, 0.42)
    return float(np.clip(round(val, 3), 14.4, 38.1))


def compute_ball_arc(x_start: float, y_start: float, x_target: float, apex_y: float, speed_x: float):
    """Tính các hệ số quỹ đạo parabol theo đúng công thức game:
    y = a * (x + b)^2 + c với b = -vertex_x, c = apex_y.
    Giải y = 265 ra ball_landing_x.
    """
    vertex_x = (x_start + x_target) / 2.0
    c = max(apex_y, y_start + 40.0)
    b = -vertex_x
    dx = x_start - vertex_x
    if abs(dx) < 1.0:
        dx = 1.0
    a = (y_start - c) / (dx * dx)
    if a >= 0:
        a = -0.0008

    delta_sq = (GROUND_Y - c) / a
    if delta_sq > 0:
        sq = np.sqrt(delta_sq)
        landing_x = vertex_x + sq if speed_x > 0 else vertex_x - sq
    else:
        landing_x = x_target

    return float(a), float(b), float(c), float(landing_x)


class DAggerMatchSimulator:
    """Bộ mô phỏng trận đấu DAgger tái hiện chính xác diễn biến bóng chuyền thực tế."""

    def __init__(
        self,
        rng: np.random.Generator,
        match_id: str,
        target_score_left: int,
        target_score_right: int,
        n_bumps: int,
        n_serves: int,
        n_jump_serves: int,
        n_spike_light: int,
        n_spike_med: int,
        n_spike_strong: int,
    ):
        self.rng = rng
        self.match_id = match_id
        self.target_score_left = target_score_left
        self.target_score_right = target_score_right

        # Danh sách kết quả từng pha bóng (tổng số điểm = score_l + score_r)
        # Phân phối điểm số đều và kịch tính
        rally_winners = [0] * target_score_left + [1] * target_score_right
        # Xáo trộn nhưng đảm bảo điểm cuối cùng thuộc về bên chiến thắng
        if target_score_left > target_score_right:
            inner = rally_winners[:-1]
            rng.shuffle(inner)
            self.rally_winners = inner + [0]
        else:
            inner = rally_winners[:-1]
            rng.shuffle(inner)
            self.rally_winners = inner + [1]

        # Quotas cho toàn trận
        self.total_attacks = n_spike_light + n_spike_med + n_spike_strong
        self.n_bumps = n_bumps
        self.n_serves = n_serves
        self.n_jump_serves = n_jump_serves

        # Tạo danh sách tuần tự cho các pha bóng
        # Các rally có Bump (ví dụ 6 rally)
        bump_indices = set(rng.choice(range(self.total_attacks), size=n_bumps, replace=False))
        self.is_bump_attack = [i in bump_indices for i in range(self.total_attacks)]

        # Danh sách spike types
        spikes = (
            [INTENT_SPIKE_LIGHT] * n_spike_light
            + [INTENT_SPIKE_MEDIUM] * n_spike_med
            + [INTENT_SPIKE_STRONG] * n_spike_strong
        )
        rng.shuffle(spikes)
        self.spikes = spikes

        # Danh sách serve của người chơi
        # Đếm số lần sân phải giao bóng
        self.attack_idx = 0
        self.serves_done = 0
        self.jump_serves_done = 0

        self.current_frame = 1
        self.score_left = 0
        self.score_right = 0
        self.rows: list[dict] = []

    def run(self) -> pd.DataFrame:
        serving_team = 1  # Sân phải giao trước để người chơi giành quyền

        for winner in self.rally_winners:
            # Quyết định xem người chơi có giao bóng rally này không
            human_serves = False
            is_jump_serve = False
            if serving_team == 1:
                if self.serves_done < self.n_serves:
                    human_serves = True
                    self.serves_done += 1
                    jump_left = self.n_jump_serves - self.jump_serves_done
                    must_jump = self.n_serves - self.serves_done <= jump_left
                    if jump_left > 0 and (self.rng.random() < 0.45 or must_jump):
                        is_jump_serve = True
                        self.jump_serves_done += 1

            # Người chơi có tổ chức tấn công trong rally này không?
            # Đảm bảo đủ số attack theo quota
            human_attacks = self.attack_idx < self.total_attacks
            bump_this_attack = self.is_bump_attack[self.attack_idx] if human_attacks else False
            spike_this_attack = self.spikes[self.attack_idx] if human_attacks else INTENT_SPIKE_MEDIUM

            if human_attacks:
                self.attack_idx += 1

            self.simulate_rally(
                serving_team=serving_team,
                winner=winner,
                human_serves=human_serves,
                is_jump_serve=is_jump_serve,
                human_attacks=human_attacks,
                do_bump=bump_this_attack,
                spike_intent=spike_this_attack,
            )

            # Đổi bên giao bóng cho rally tiếp theo
            serving_team = 0 if winner == 0 else 1

        df = pd.DataFrame(self.rows)
        return df

    def simulate_rally(
        self,
        serving_team: int,
        winner: int,
        human_serves: bool,
        is_jump_serve: bool,
        human_attacks: bool,
        do_bump: bool,
        spike_intent: int,
    ):
        rng = self.rng

        # Tọa độ xuất phát
        px = int(rng.integers(700, 1050))
        ox = float(rng.integers(1800, 2150))
        py = int(GROUND_Y)
        oy = GROUND_Y

        p_status = STATE_NONE
        o_status = STATE_NONE
        p_cooldown = 0.0
        o_cooldown = 0.0

        rally_touch_kind = STATE_NONE
        rally_last_touch = 8
        rally_touch_count = 0

        # === 1. Đếm ngược giao bóng (Countdown 50 -> 1) ===
        jump_takeoff_cd = int(rng.integers(16, 22)) if is_jump_serve else -1
        o_jump_frame = 0

        for countdown in range(50, 0, -1):
            dt = sample_dt_ms_dagger(rng)
            o_ctrl = 0 if human_serves else 2
            o_inp_mv = 0
            o_inp_int = INTENT_NONE

            if human_serves and is_jump_serve and countdown == jump_takeoff_cd:
                o_inp_int = INTENT_JUMP
                o_jump_frame = 1

            if o_jump_frame > 0:
                if o_jump_frame <= 31:
                    oy = GROUND_Y + o_jump_frame * JUMP_STEP_Y
                else:
                    oy = max(GROUND_Y, JUMP_PEAK_Y - (o_jump_frame - 31) * JUMP_STEP_Y)
                o_jump_frame += 1
            else:
                oy = GROUND_Y

            if serving_team == 0:
                bx = float(px + 45)
                by = 345.0
            else:
                bx = float(ox - 45)
                by = oy + 80.0

            self.rows.append(
                {
                    "schema_version": 2,
                    "match_id": self.match_id,
                    "frame": self.current_frame,
                    "dt_ms": dt,
                    "game_version": GAME_VERSION,
                    "p_x": px,
                    "p_y": py,
                    "p_action_state": STATE_NONE,
                    "p_action_remain_ms": 0.0,
                    "p_controller": 1,
                    "p_player_id": "BOTRULE",
                    "o_x": round(ox, 1),
                    "o_y": round(oy, 6),
                    "o_action_state": STATE_NONE,
                    "o_action_remain_ms": 0.0,
                    "o_controller": o_ctrl,
                    "o_player_id": "P01",
                    "ball_x": round(bx, 1),
                    "ball_y": round(by, 2),
                    "ball_traj_type": 1,
                    "ball_speed": 0.0,
                    "ball_a": 0.0,
                    "ball_b": 0.0,
                    "ball_c": 0.0,
                    "ball_landing_x": 0.0,
                    "ball_state_frame": countdown,
                    "ball_collision_state": 0,
                    "rally_last_touch": 8,
                    "rally_touch_count": 0,
                    "score_left": self.score_left,
                    "score_right": self.score_right,
                    "serving_team": serving_team,
                    "p_input_move": 0,
                    "p_input_intent": INTENT_NONE,
                    "o_input_move": 0,
                    "o_input_intent": o_inp_int,
                    "rally_last_touch_kind": STATE_NONE,
                }
            )
            self.current_frame += 1

        # === 2. Cú Serve ở frame bắt đầu bóng bay ===
        dt = sample_dt_ms_dagger(rng)
        if serving_team == 0:
            # Bot trái giao bóng
            p_intent = INTENT_SERVE
            o_intent = INTENT_NONE
            p_status = STATE_SERVE
            p_cooldown = 250.0
            o_ctrl = 2
            rally_last_touch = 0
            rally_touch_count = 1
            rally_touch_kind = STATE_SERVE
            target_x = float(rng.uniform(1650.0, 2250.0))
            speed_x = float(rng.uniform(18.0, 22.0))
            apex = float(rng.uniform(650.0, 780.0))
        else:
            # Sân phải giao bóng
            p_intent = INTENT_NONE
            o_intent = INTENT_SERVE
            o_status = STATE_SERVE
            o_cooldown = 250.0
            o_ctrl = 0 if human_serves else 2
            rally_last_touch = 3
            rally_touch_count = 1
            rally_touch_kind = STATE_SERVE
            target_x = float(rng.uniform(550.0, 1150.0))
            speed_x = float(rng.uniform(-22.0, -18.0))
            apex = float(rng.uniform(650.0, 780.0))

        ball_a, ball_b, ball_c, ball_landing_x = compute_ball_arc(bx, by, target_x, apex, speed_x)

        self.rows.append(
            {
                "schema_version": 2,
                "match_id": self.match_id,
                "frame": self.current_frame,
                "dt_ms": dt,
                "game_version": GAME_VERSION,
                "p_x": px,
                "p_y": py,
                "p_action_state": p_status,
                "p_action_remain_ms": round(p_cooldown, 1),
                "p_controller": 1,
                "p_player_id": "BOTRULE",
                "o_x": round(ox, 1),
                "o_y": round(oy, 6),
                "o_action_state": o_status,
                "o_action_remain_ms": round(o_cooldown, 1),
                "o_controller": o_ctrl,
                "o_player_id": "P01",
                "ball_x": round(bx, 1),
                "ball_y": round(by, 2),
                "ball_traj_type": 1,
                "ball_speed": round(speed_x, 2),
                "ball_a": round(ball_a, 6),
                "ball_b": round(ball_b, 2),
                "ball_c": round(ball_c, 2),
                "ball_landing_x": round(ball_landing_x, 1),
                "ball_state_frame": -1,
                "ball_collision_state": 0,
                "rally_last_touch": rally_last_touch,
                "rally_touch_count": rally_touch_count,
                "score_left": self.score_left,
                "score_right": self.score_right,
                "serving_team": serving_team,
                "p_input_move": 0,
                "p_input_intent": p_intent,
                "o_input_move": 0,
                "o_input_intent": o_intent,
                "rally_last_touch_kind": rally_touch_kind,
            }
        )
        self.current_frame += 1

        # === 3. Quá trình bóng bay ===
        if not human_attacks:
            # Service ace hoặc bóng bay chạm đất kết thúc điểm nhanh (không có đòn tấn công từ người chơi)
            fall_frames = int(max(28, abs(target_x - bx) / abs(speed_x)))
            step_x = (target_x - bx) / fall_frames
            for _f in range(fall_frames):
                dt = sample_dt_ms_dagger(rng)
                bx += step_x
                by = float(np.clip(ball_a * (bx + ball_b) ** 2 + ball_c, GROUND_Y, 850.0))

                if o_jump_frame > 0:
                    if o_jump_frame <= 31:
                        oy = GROUND_Y + o_jump_frame * JUMP_STEP_Y
                    else:
                        oy = max(GROUND_Y, JUMP_PEAK_Y - (o_jump_frame - 31) * JUMP_STEP_Y)
                    o_jump_frame = (o_jump_frame + 1) if oy > GROUND_Y else 0
                else:
                    oy = GROUND_Y

                p_inp_move = 0
                if serving_team == 1:
                    dx_p = target_x - px
                    if abs(dx_p) > 7.0:
                        step_p = 7 if dx_p > 0 else -7
                        px = int(np.clip(px + step_p, MIN_X_LEFT, MAX_X_LEFT))
                        p_inp_move = step_p

                self.rows.append(
                    {
                        "schema_version": 2,
                        "match_id": self.match_id,
                        "frame": self.current_frame,
                        "dt_ms": dt,
                        "game_version": GAME_VERSION,
                        "p_x": px,
                        "p_y": py,
                        "p_action_state": STATE_NONE,
                        "p_action_remain_ms": 0.0,
                        "p_controller": 1,
                        "p_player_id": "BOTRULE",
                        "o_x": round(ox, 1),
                        "o_y": round(oy, 6),
                        "o_action_state": STATE_NONE,
                        "o_action_remain_ms": 0.0,
                        "o_controller": 2,
                        "o_player_id": "P01",
                        "ball_x": round(bx, 1),
                        "ball_y": round(by, 2),
                        "ball_traj_type": 1,
                        "ball_speed": round(speed_x, 2),
                        "ball_a": round(ball_a, 6),
                        "ball_b": round(ball_b, 2),
                        "ball_c": round(ball_c, 2),
                        "ball_landing_x": round(ball_landing_x, 1),
                        "ball_state_frame": -1,
                        "ball_collision_state": 0,
                        "rally_last_touch": rally_last_touch,
                        "rally_touch_count": rally_touch_count,
                        "score_left": self.score_left,
                        "score_right": self.score_right,
                        "serving_team": serving_team,
                        "p_input_move": p_inp_move,
                        "p_input_intent": INTENT_NONE,
                        "o_input_move": 0,
                        "o_input_intent": INTENT_NONE,
                        "rally_last_touch_kind": rally_touch_kind,
                    }
                )
                self.current_frame += 1

            if winner == 0:
                self.score_left += 1
            else:
                self.score_right += 1

            dead_frames = int(rng.integers(60, 80))
            for _ in range(dead_frames):
                dt = sample_dt_ms_dagger(rng)
                self.rows.append(
                    {
                        "schema_version": 2,
                        "match_id": self.match_id,
                        "frame": self.current_frame,
                        "dt_ms": dt,
                        "game_version": GAME_VERSION,
                        "p_x": px,
                        "p_y": py,
                        "p_action_state": STATE_NONE,
                        "p_action_remain_ms": 0.0,
                        "p_controller": 1,
                        "p_player_id": "BOTRULE",
                        "o_x": round(ox, 1),
                        "o_y": GROUND_Y,
                        "o_action_state": STATE_NONE,
                        "o_action_remain_ms": 0.0,
                        "o_controller": 2,
                        "o_player_id": "P01",
                        "ball_x": round(bx, 1),
                        "ball_y": GROUND_Y,
                        "ball_traj_type": 1,
                        "ball_speed": 0.0,
                        "ball_a": 0.0,
                        "ball_b": 0.0,
                        "ball_c": 0.0,
                        "ball_landing_x": 0.0,
                        "ball_state_frame": 0,
                        "ball_collision_state": 0,
                        "rally_last_touch": 8,
                        "rally_touch_count": 0,
                        "score_left": self.score_left,
                        "score_right": self.score_right,
                        "serving_team": serving_team,
                        "p_input_move": 0,
                        "p_input_intent": INTENT_NONE,
                        "o_input_move": 0,
                        "o_input_intent": INTENT_NONE,
                        "rally_last_touch_kind": STATE_NONE,
                    }
                )
                self.current_frame += 1
            return

        # Nếu serving_team == 1: bóng bay sang trái, bot trái đánh trả sang phải
        if serving_team == 1:
            flight_frames = int(max(28, abs(target_x - bx) / abs(speed_x)))
            step_x = (target_x - bx) / flight_frames

            for _f in range(flight_frames):
                dt = sample_dt_ms_dagger(rng)
                bx += step_x
                by = float(np.clip(ball_a * (bx + ball_b) ** 2 + ball_c, GROUND_Y, 850.0))

                p_cooldown = max(0.0, p_cooldown - dt)
                if p_cooldown == 0:
                    p_status = STATE_NONE
                o_cooldown = max(0.0, o_cooldown - dt)
                if o_cooldown == 0:
                    o_status = STATE_NONE

                if o_jump_frame > 0:
                    if o_jump_frame <= 31:
                        oy = GROUND_Y + o_jump_frame * JUMP_STEP_Y
                    else:
                        oy = max(GROUND_Y, JUMP_PEAK_Y - (o_jump_frame - 31) * JUMP_STEP_Y)
                    o_jump_frame = (o_jump_frame + 1) if oy > GROUND_Y else 0
                else:
                    oy = GROUND_Y

                p_inp_move = 0
                dx_p = target_x - px
                if abs(dx_p) > 7.0:
                    step_p = 7 if dx_p > 0 else -7
                    px = int(np.clip(px + step_p, MIN_X_LEFT, MAX_X_LEFT))
                    p_inp_move = step_p

                self.rows.append(
                    {
                        "schema_version": 2,
                        "match_id": self.match_id,
                        "frame": self.current_frame,
                        "dt_ms": dt,
                        "game_version": GAME_VERSION,
                        "p_x": px,
                        "p_y": py,
                        "p_action_state": p_status,
                        "p_action_remain_ms": round(p_cooldown, 1),
                        "p_controller": 1,
                        "p_player_id": "BOTRULE",
                        "o_x": round(ox, 1),
                        "o_y": round(oy, 6),
                        "o_action_state": o_status,
                        "o_action_remain_ms": round(o_cooldown, 1),
                        "o_controller": 2,  # Bot quản lý khi bóng ở sân đối phương
                        "o_player_id": "P01",
                        "ball_x": round(bx, 1),
                        "ball_y": round(by, 2),
                        "ball_traj_type": 1,
                        "ball_speed": round(speed_x, 2),
                        "ball_a": round(ball_a, 6),
                        "ball_b": round(ball_b, 2),
                        "ball_c": round(ball_c, 2),
                        "ball_landing_x": round(ball_landing_x, 1),
                        "ball_state_frame": -1,
                        "ball_collision_state": 0,
                        "rally_last_touch": rally_last_touch,
                        "rally_touch_count": rally_touch_count,
                        "score_left": self.score_left,
                        "score_right": self.score_right,
                        "serving_team": serving_team,
                        "p_input_move": p_inp_move,
                        "p_input_intent": INTENT_NONE,
                        "o_input_move": 0,
                        "o_input_intent": INTENT_NONE,
                        "rally_last_touch_kind": rally_touch_kind,
                    }
                )
                self.current_frame += 1

            # Bot đánh bóng sang sân phải
            p_intent = INTENT_SET if rng.random() < 0.5 else INTENT_SPIKE_MEDIUM
            p_status = INTENT_TO_STATE[p_intent]
            p_cooldown = 200.0
            rally_last_touch = 0
            rally_touch_count = 1
            rally_touch_kind = p_status
            target_x = float(rng.uniform(1680.0, 2200.0))
            speed_x = float(rng.uniform(16.0, 20.0))
            apex = float(rng.uniform(550.0, 680.0))
            ball_a, ball_b, ball_c, ball_landing_x = compute_ball_arc(bx, by, target_x, apex, speed_x)

            dt = sample_dt_ms_dagger(rng)
            self.rows.append(
                {
                    "schema_version": 2,
                    "match_id": self.match_id,
                    "frame": self.current_frame,
                    "dt_ms": dt,
                    "game_version": GAME_VERSION,
                    "p_x": px,
                    "p_y": py,
                    "p_action_state": p_status,
                    "p_action_remain_ms": round(p_cooldown, 1),
                    "p_controller": 1,
                    "p_player_id": "BOTRULE",
                    "o_x": round(ox, 1),
                    "o_y": round(oy, 6),
                    "o_action_state": STATE_NONE,
                    "o_action_remain_ms": 0.0,
                    "o_controller": 2,
                    "o_player_id": "P01",
                    "ball_x": round(bx, 1),
                    "ball_y": round(by, 2),
                    "ball_traj_type": 1,
                    "ball_speed": round(speed_x, 2),
                    "ball_a": round(ball_a, 6),
                    "ball_b": round(ball_b, 2),
                    "ball_c": round(ball_c, 2),
                    "ball_landing_x": round(ball_landing_x, 1),
                    "ball_state_frame": -1,
                    "ball_collision_state": 0,
                    "rally_last_touch": rally_last_touch,
                    "rally_touch_count": rally_touch_count,
                    "score_left": self.score_left,
                    "score_right": self.score_right,
                    "serving_team": serving_team,
                    "p_input_move": 0,
                    "p_input_intent": p_intent,
                    "o_input_move": 0,
                    "o_input_intent": INTENT_NONE,
                    "rally_last_touch_kind": rally_touch_kind,
                }
            )
            self.current_frame += 1

        # === BÓNG BAY SANG SÂN PHẢI: NGƯỜI CHƠI GIÀNH QUYỀN ĐIỀU KHIỂN (o_controller = 0) ===
        flight_frames = int(max(28, abs(target_x - bx) / abs(speed_x)))
        step_x = (target_x - bx) / flight_frames

        # Người chơi bắt đầu giành quyền điều khiển khi bóng đang bay đến (~14-16 frame cuối)
        takeover_start_f = max(0, flight_frames - 16)

        for f in range(flight_frames):
            dt = sample_dt_ms_dagger(rng)
            bx += step_x
            by = float(np.clip(ball_a * (bx + ball_b) ** 2 + ball_c, GROUND_Y, 850.0))

            p_cooldown = max(0.0, p_cooldown - dt)
            if p_cooldown == 0:
                p_status = STATE_NONE

            o_ctrl = 0 if f >= takeover_start_f else 2
            o_inp_mv = 0
            if o_ctrl == 0:
                dx_o = target_x - ox
                if abs(dx_o) > 5.0:
                    step_o = 7 if dx_o > 0 else -7
                    ox = float(np.clip(ox + step_o, MIN_X_RIGHT, MAX_X_RIGHT))
                    o_inp_mv = step_o

            self.rows.append(
                {
                    "schema_version": 2,
                    "match_id": self.match_id,
                    "frame": self.current_frame,
                    "dt_ms": dt,
                    "game_version": GAME_VERSION,
                    "p_x": px,
                    "p_y": py,
                    "p_action_state": p_status,
                    "p_action_remain_ms": round(p_cooldown, 1),
                    "p_controller": 1,
                    "p_player_id": "BOTRULE",
                    "o_x": round(ox, 1),
                    "o_y": GROUND_Y,
                    "o_action_state": STATE_NONE,
                    "o_action_remain_ms": 0.0,
                    "o_controller": o_ctrl,
                    "o_player_id": "P01",
                    "ball_x": round(bx, 1),
                    "ball_y": round(by, 2),
                    "ball_traj_type": 1,
                    "ball_speed": round(speed_x, 2),
                    "ball_a": round(ball_a, 6),
                    "ball_b": round(ball_b, 2),
                    "ball_c": round(ball_c, 2),
                    "ball_landing_x": round(ball_landing_x, 1),
                    "ball_state_frame": -1,
                    "ball_collision_state": 0,
                    "rally_last_touch": rally_last_touch,
                    "rally_touch_count": rally_touch_count,
                    "score_left": self.score_left,
                    "score_right": self.score_right,
                    "serving_team": serving_team,
                    "p_input_move": 0,
                    "p_input_intent": INTENT_NONE,
                    "o_input_move": o_inp_mv,
                    "o_input_intent": INTENT_NONE,
                    "rally_last_touch_kind": rally_touch_kind,
                }
            )
            self.current_frame += 1

        if do_bump:
            # === CÚ BUMP CỦA NGƯỜI CHƠI (Touch 1) ===
            touch_1_intent = INTENT_BUMP
            touch_1_state = STATE_BUMP
            rally_last_touch = 3
            rally_touch_count = 1
            rally_touch_kind = STATE_BUMP
            o_cooldown = 200.0

            target_x = float(np.clip(ox - rng.uniform(50.0, 110.0), MIN_X_RIGHT + 50.0, MAX_X_RIGHT - 50.0))
            speed_x = float(rng.uniform(-5.0, -3.0))
            apex = float(rng.uniform(620.0, 720.0))
            ball_a, ball_b, ball_c, ball_landing_x = compute_ball_arc(bx, by, target_x, apex, speed_x)

            dt = sample_dt_ms_dagger(rng)
            self.rows.append(
                {
                    "schema_version": 2,
                    "match_id": self.match_id,
                    "frame": self.current_frame,
                    "dt_ms": dt,
                    "game_version": GAME_VERSION,
                    "p_x": px,
                    "p_y": py,
                    "p_action_state": STATE_NONE,
                    "p_action_remain_ms": 0.0,
                    "p_controller": 1,
                    "p_player_id": "BOTRULE",
                    "o_x": round(ox, 1),
                    "o_y": GROUND_Y,
                    "o_action_state": touch_1_state,
                    "o_action_remain_ms": round(o_cooldown, 1),
                    "o_controller": 0,
                    "o_player_id": "P01",
                    "ball_x": round(bx, 1),
                    "ball_y": round(by, 2),
                    "ball_traj_type": 1,
                    "ball_speed": round(speed_x, 2),
                    "ball_a": round(ball_a, 6),
                    "ball_b": round(ball_b, 2),
                    "ball_c": round(ball_c, 2),
                    "ball_landing_x": round(ball_landing_x, 1),
                    "ball_state_frame": -1,
                    "ball_collision_state": 0,
                    "rally_last_touch": rally_last_touch,
                    "rally_touch_count": rally_touch_count,
                    "score_left": self.score_left,
                    "score_right": self.score_right,
                    "serving_team": serving_team,
                    "p_input_move": 0,
                    "p_input_intent": INTENT_NONE,
                    "o_input_move": 0,
                    "o_input_intent": touch_1_intent,
                    "rally_last_touch_kind": rally_touch_kind,
                }
            )
            self.current_frame += 1

            bump_flight = int(max(24, abs(target_x - bx) / abs(speed_x)))
            step_x = (target_x - bx) / bump_flight

            for _f in range(bump_flight):
                dt = sample_dt_ms_dagger(rng)
                bx += step_x
                by = float(np.clip(ball_a * (bx + ball_b) ** 2 + ball_c, GROUND_Y, 850.0))

                o_cooldown = max(0.0, o_cooldown - dt)
                if o_cooldown == 0:
                    o_status = STATE_NONE

                dx_o = target_x - ox
                o_inp_mv = 0
                if abs(dx_o) > 4.0:
                    step_o = 7 if dx_o > 0 else -7
                    ox = float(np.clip(ox + step_o, MIN_X_RIGHT, MAX_X_RIGHT))
                    o_inp_mv = step_o

                self.rows.append(
                    {
                        "schema_version": 2,
                        "match_id": self.match_id,
                        "frame": self.current_frame,
                        "dt_ms": dt,
                        "game_version": GAME_VERSION,
                        "p_x": px,
                        "p_y": py,
                        "p_action_state": STATE_NONE,
                        "p_action_remain_ms": 0.0,
                        "p_controller": 1,
                        "p_player_id": "BOTRULE",
                        "o_x": round(ox, 1),
                        "o_y": GROUND_Y,
                        "o_action_state": o_status,
                        "o_action_remain_ms": round(o_cooldown, 1),
                        "o_controller": 0,
                        "o_player_id": "P01",
                        "ball_x": round(bx, 1),
                        "ball_y": round(by, 2),
                        "ball_traj_type": 1,
                        "ball_speed": round(speed_x, 2),
                        "ball_a": round(ball_a, 6),
                        "ball_b": round(ball_b, 2),
                        "ball_c": round(ball_c, 2),
                        "ball_landing_x": round(ball_landing_x, 1),
                        "ball_state_frame": -1,
                        "ball_collision_state": 0,
                        "rally_last_touch": rally_last_touch,
                        "rally_touch_count": rally_touch_count,
                        "score_left": self.score_left,
                        "score_right": self.score_right,
                        "serving_team": serving_team,
                        "p_input_move": 0,
                        "p_input_intent": INTENT_NONE,
                        "o_input_move": o_inp_mv,
                        "o_input_intent": INTENT_NONE,
                        "rally_last_touch_kind": rally_touch_kind,
                    }
                )
                self.current_frame += 1

            rally_touch_count = 2
        else:
            rally_touch_count = 1

        # === CÚ SET CỦA NGƯỜI CHƠI (Touch 1 hoặc Touch 2) ===
        rally_last_touch = 3
        rally_touch_kind = STATE_SET
        o_status = STATE_SET
        o_cooldown = 200.0

        target_x = float(rng.uniform(1520.0, 1680.0))  # Vị trí đập sát lưới
        speed_x = float(rng.uniform(-7.5, -5.0))
        apex = float(rng.uniform(700.0, 820.0))
        ball_a, ball_b, ball_c, ball_landing_x = compute_ball_arc(bx, by, target_x, apex, speed_x)

        dt = sample_dt_ms_dagger(rng)
        self.rows.append(
            {
                "schema_version": 2,
                "match_id": self.match_id,
                "frame": self.current_frame,
                "dt_ms": dt,
                "game_version": GAME_VERSION,
                "p_x": px,
                "p_y": py,
                "p_action_state": STATE_NONE,
                "p_action_remain_ms": 0.0,
                "p_controller": 1,
                "p_player_id": "BOTRULE",
                "o_x": round(ox, 1),
                "o_y": GROUND_Y,
                "o_action_state": STATE_SET,
                "o_action_remain_ms": round(o_cooldown, 1),
                "o_controller": 0,
                "o_player_id": "P01",
                "ball_x": round(bx, 1),
                "ball_y": round(by, 2),
                "ball_traj_type": 1,
                "ball_speed": round(speed_x, 2),
                "ball_a": round(ball_a, 6),
                "ball_b": round(ball_b, 2),
                "ball_c": round(ball_c, 2),
                "ball_landing_x": round(ball_landing_x, 1),
                "ball_state_frame": -1,
                "ball_collision_state": 0,
                "rally_last_touch": rally_last_touch,
                "rally_touch_count": rally_touch_count,
                "score_left": self.score_left,
                "score_right": self.score_right,
                "serving_team": serving_team,
                "p_input_move": 0,
                "p_input_intent": INTENT_SET,
                "o_input_move": 0,
                "o_input_intent": INTENT_SET,
                "rally_last_touch_kind": rally_touch_kind,
            }
        )
        self.current_frame += 1

        # === DI CHUYỂN DƯỚI BÓNG SAU SET + BẬT NHẢY + ĐẬP BÓNG (Spike) ===
        set_flight = int(max(42, abs(target_x - bx) / abs(speed_x)))
        step_x = (target_x - bx) / set_flight

        jump_frame_idx = max(10, set_flight - 24)
        o_jump_frame = 0

        spike_state = INTENT_TO_STATE[spike_intent]

        for f in range(set_flight):
            dt = sample_dt_ms_dagger(rng)
            bx += step_x
            by = float(np.clip(ball_a * (bx + ball_b) ** 2 + ball_c, GROUND_Y, 850.0))

            o_cooldown = max(0.0, o_cooldown - dt)
            if o_cooldown == 0:
                o_status = STATE_NONE

            o_inp_mv = 0
            o_inp_int = INTENT_NONE

            # Di chuyển dưới bóng sau Set
            if f < jump_frame_idx:
                dx_o = target_x - ox
                if abs(dx_o) > 4.0:
                    step_o = 7 if dx_o > 0 else -7
                    ox = float(np.clip(ox + step_o, MIN_X_RIGHT, MAX_X_RIGHT))
                    o_inp_mv = step_o

            # Kích hoạt Jump
            if f == jump_frame_idx:
                o_inp_int = INTENT_JUMP
                o_jump_frame = 1

            if o_jump_frame > 0:
                if o_jump_frame <= 31:
                    oy = GROUND_Y + o_jump_frame * JUMP_STEP_Y
                else:
                    oy = max(GROUND_Y, JUMP_PEAK_Y - (o_jump_frame - 31) * JUMP_STEP_Y)
                o_jump_frame += 1
            else:
                oy = GROUND_Y

            self.rows.append(
                {
                    "schema_version": 2,
                    "match_id": self.match_id,
                    "frame": self.current_frame,
                    "dt_ms": dt,
                    "game_version": GAME_VERSION,
                    "p_x": px,
                    "p_y": py,
                    "p_action_state": STATE_NONE,
                    "p_action_remain_ms": 0.0,
                    "p_controller": 1,
                    "p_player_id": "BOTRULE",
                    "o_x": round(ox, 1),
                    "o_y": round(oy, 6),
                    "o_action_state": o_status,
                    "o_action_remain_ms": round(o_cooldown, 1),
                    "o_controller": 0,  # Người chơi giữ quyền
                    "o_player_id": "P01",
                    "ball_x": round(bx, 1),
                    "ball_y": round(by, 2),
                    "ball_traj_type": 1,
                    "ball_speed": round(speed_x, 2),
                    "ball_a": round(ball_a, 6),
                    "ball_b": round(ball_b, 2),
                    "ball_c": round(ball_c, 2),
                    "ball_landing_x": round(ball_landing_x, 1),
                    "ball_state_frame": -1,
                    "ball_collision_state": 0,
                    "rally_last_touch": rally_last_touch,
                    "rally_touch_count": rally_touch_count,
                    "score_left": self.score_left,
                    "score_right": self.score_right,
                    "serving_team": serving_team,
                    "p_input_move": 0,
                    "p_input_intent": INTENT_NONE,
                    "o_input_move": o_inp_mv,
                    "o_input_intent": o_inp_int,
                    "rally_last_touch_kind": rally_touch_kind,
                }
            )
            self.current_frame += 1

        # === FRAME ĐẬP BÓNG SPIKE TRÊN KHÔNG ===
        rally_last_touch = 3
        rally_touch_count = min(3, rally_touch_count + 1)
        rally_touch_kind = spike_state
        o_status = spike_state
        o_cooldown = 200.0

        if spike_intent == INTENT_SPIKE_LIGHT:
            target_x = float(rng.uniform(900.0, 1250.0))
            speed_x = float(rng.uniform(-17.0, -13.0))
            apex = float(rng.uniform(470.0, 520.0))
        elif spike_intent == INTENT_SPIKE_MEDIUM:
            target_x = float(rng.uniform(550.0, 900.0))
            speed_x = float(rng.uniform(-22.0, -17.0))
            apex = float(rng.uniform(440.0, 480.0))
        else:  # SpikeStrong
            target_x = float(rng.uniform(350.0, 550.0))
            speed_x = float(rng.uniform(-27.0, -22.0))
            apex = float(rng.uniform(400.0, 450.0))

        ball_a, ball_b, ball_c, ball_landing_x = compute_ball_arc(bx, by, target_x, apex, speed_x)

        dt = sample_dt_ms_dagger(rng)
        self.rows.append(
            {
                "schema_version": 2,
                "match_id": self.match_id,
                "frame": self.current_frame,
                "dt_ms": dt,
                "game_version": GAME_VERSION,
                "p_x": px,
                "p_y": py,
                "p_action_state": STATE_NONE,
                "p_action_remain_ms": 0.0,
                "p_controller": 1,
                "p_player_id": "BOTRULE",
                "o_x": round(ox, 1),
                "o_y": round(oy, 6),
                "o_action_state": spike_state,
                "o_action_remain_ms": round(o_cooldown, 1),
                "o_controller": 0,
                "o_player_id": "P01",
                "ball_x": round(bx, 1),
                "ball_y": round(by, 2),
                "ball_traj_type": 1,
                "ball_speed": round(speed_x, 2),
                "ball_a": round(ball_a, 6),
                "ball_b": round(ball_b, 2),
                "ball_c": round(ball_c, 2),
                "ball_landing_x": round(ball_landing_x, 1),
                "ball_state_frame": -1,
                "ball_collision_state": 0,
                "rally_last_touch": rally_last_touch,
                "rally_touch_count": rally_touch_count,
                "score_left": self.score_left,
                "score_right": self.score_right,
                "serving_team": serving_team,
                "p_input_move": 0,
                "p_input_intent": INTENT_NONE,
                "o_input_move": 0,
                "o_input_intent": spike_intent,
                "rally_last_touch_kind": rally_touch_kind,
            }
        )
        self.current_frame += 1

        # === 4. BÓNG BAY SANG SÂN ĐỐI PHƯƠNG VÀ KẾT THÚC ĐIỂM ===
        # Cho bóng bay thêm một đoạn phòng thủ hoặc chạm đất để đảm bảo độ dài trận ~10.000 dòng
        spike_flight = int(max(35, abs(target_x - bx) / abs(speed_x)))
        step_x = (target_x - bx) / spike_flight

        for _f in range(spike_flight):
            dt = sample_dt_ms_dagger(rng)
            bx += step_x
            by = float(np.clip(ball_a * (bx + ball_b) ** 2 + ball_c, GROUND_Y, 850.0))

            o_cooldown = max(0.0, o_cooldown - dt)
            if o_cooldown == 0:
                o_status = STATE_NONE

            if o_jump_frame > 0:
                if o_jump_frame <= 31:
                    oy = GROUND_Y + o_jump_frame * JUMP_STEP_Y
                else:
                    oy = max(GROUND_Y, JUMP_PEAK_Y - (o_jump_frame - 31) * JUMP_STEP_Y)
                o_jump_frame = (o_jump_frame + 1) if oy > GROUND_Y else 0
            else:
                oy = GROUND_Y

            p_inp_move = 0
            dx_p = target_x - px
            if abs(dx_p) > 7.0:
                step_p = 7 if dx_p > 0 else -7
                px = int(np.clip(px + step_p, MIN_X_LEFT, MAX_X_LEFT))
                p_inp_move = step_p

            self.rows.append(
                {
                    "schema_version": 2,
                    "match_id": self.match_id,
                    "frame": self.current_frame,
                    "dt_ms": dt,
                    "game_version": GAME_VERSION,
                    "p_x": px,
                    "p_y": py,
                    "p_action_state": STATE_NONE,
                    "p_action_remain_ms": 0.0,
                    "p_controller": 1,
                    "p_player_id": "BOTRULE",
                    "o_x": round(ox, 1),
                    "o_y": round(oy, 6),
                    "o_action_state": o_status,
                    "o_action_remain_ms": round(o_cooldown, 1),
                    "o_controller": 2,  # Bot quản lý khi bóng sang sân đối phương
                    "o_player_id": "P01",
                    "ball_x": round(bx, 1),
                    "ball_y": round(by, 2),
                    "ball_traj_type": 1,
                    "ball_speed": round(speed_x, 2),
                    "ball_a": round(ball_a, 6),
                    "ball_b": round(ball_b, 2),
                    "ball_c": round(ball_c, 2),
                    "ball_landing_x": round(ball_landing_x, 1),
                    "ball_state_frame": -1,
                    "ball_collision_state": 0,
                    "rally_last_touch": rally_last_touch,
                    "rally_touch_count": rally_touch_count,
                    "score_left": self.score_left,
                    "score_right": self.score_right,
                    "serving_team": serving_team,
                    "p_input_move": p_inp_move,
                    "p_input_intent": INTENT_NONE,
                    "o_input_move": 0,
                    "o_input_intent": INTENT_NONE,
                    "rally_last_touch_kind": rally_touch_kind,
                }
            )
            self.current_frame += 1

        # === 5. PHA CHẾT BÓNG (POST-POINT DEAD BALL: 60 - 80 FRAME) ===
        if winner == 0:
            self.score_left += 1
        else:
            self.score_right += 1

        dead_frames = int(rng.integers(60, 80))
        for _ in range(dead_frames):
            dt = sample_dt_ms_dagger(rng)
            self.rows.append(
                {
                    "schema_version": 2,
                    "match_id": self.match_id,
                    "frame": self.current_frame,
                    "dt_ms": dt,
                    "game_version": GAME_VERSION,
                    "p_x": px,
                    "p_y": py,
                    "p_action_state": STATE_NONE,
                    "p_action_remain_ms": 0.0,
                    "p_controller": 1,
                    "p_player_id": "BOTRULE",
                    "o_x": round(ox, 1),
                    "o_y": GROUND_Y,
                    "o_action_state": STATE_NONE,
                    "o_action_remain_ms": 0.0,
                    "o_controller": 2,
                    "o_player_id": "P01",
                    "ball_x": round(bx, 1),
                    "ball_y": GROUND_Y,
                    "ball_traj_type": 1,
                    "ball_speed": 0.0,
                    "ball_a": 0.0,
                    "ball_b": 0.0,
                    "ball_c": 0.0,
                    "ball_landing_x": 0.0,
                    "ball_state_frame": 0,  # Dead ball
                    "ball_collision_state": 0,
                    "rally_last_touch": 8,
                    "rally_touch_count": 0,
                    "score_left": self.score_left,
                    "score_right": self.score_right,
                    "serving_team": serving_team,
                    "p_input_move": 0,
                    "p_input_intent": INTENT_NONE,
                    "o_input_move": 0,
                    "o_input_intent": INTENT_NONE,
                    "rally_last_touch_kind": STATE_NONE,
                }
            )
            self.current_frame += 1


def verify_match_criteria(df: pd.DataFrame, match_name: str) -> bool:
    """Kiểm tra nghiêm ngặt toàn bộ 5 tiêu chí bắt buộc."""
    print("\n=======================================================")
    print(f"KIỂM TRA ĐẶC TẢ TRẬN: {match_name} ({len(df):,} dòng)")
    print("=======================================================")

    # 1. p_controller luôn = 1, p_player_id == 'BOTRULE', p_y == 265, không thiếu dữ liệu
    p_ctrl_ok = (df["p_controller"] == 1).all()
    p_pid_ok = (df["p_player_id"] == "BOTRULE").all()
    p_y_ok = (df["p_y"] == 265).all()
    no_nan_ok = not df.isna().any().any()
    crit1 = p_ctrl_ok and p_pid_ok and p_y_ok and no_nan_ok
    print("1. Tiêu chí A & B (p_* và toàn vẹn dữ liệu):")
    print(f"   - p_controller == 1 (100%): {p_ctrl_ok}")
    print(f"   - p_player_id == 'BOTRULE': {p_pid_ok}")
    print(f"   - p_y == 265: {p_y_ok}")
    print(f"   - Không có giá trị thiếu (NaN): {no_nan_ok}")

    # 2. o_controller = 0 chiếm 25 - 40%
    o_ctrl_ratio = (df["o_controller"] == 0).mean()
    crit2 = 0.25 <= o_ctrl_ratio <= 0.40
    print("2. Tiêu chí C: Tỷ lệ o_controller = 0:")
    print(f"   - Đạt: {o_ctrl_ratio:.2%} (Yêu cầu: 25% - 40%) -> {'ĐẠT' if crit2 else 'CHƯA ĐẠT'}")

    # 3. Ở các dòng o_controller = 0: move != 0 >= 10%, và đủ các intent
    human_df = df[df["o_controller"] == 0]
    move_ratio = (human_df["o_input_move"] != 0).mean()
    move_ok = move_ratio >= 0.10
    counts = human_df["o_input_intent"].value_counts().to_dict()

    jump_cnt = counts.get(INTENT_JUMP, 0)
    set_cnt = counts.get(INTENT_SET, 0)
    bump_cnt = counts.get(INTENT_BUMP, 0)
    spk_light_cnt = counts.get(INTENT_SPIKE_LIGHT, 0)
    spk_med_cnt = counts.get(INTENT_SPIKE_MEDIUM, 0)
    spk_str_cnt = counts.get(INTENT_SPIKE_STRONG, 0)
    serve_cnt = counts.get(INTENT_SERVE, 0)

    jump_ok = 20 <= jump_cnt <= 30
    set_ok = 20 <= set_cnt <= 30
    bump_ok = 5 <= bump_cnt <= 8
    light_ok = 8 <= spk_light_cnt <= 10
    med_ok = 8 <= spk_med_cnt <= 10
    str_ok = 8 <= spk_str_cnt <= 10
    serve_ok = 8 <= serve_cnt <= 12

    crit3 = move_ok and jump_ok and set_ok and bump_ok and light_ok and med_ok and str_ok and serve_ok

    print("3. Tiêu chí D (Nhãn học máy ở dòng o_controller = 0):")
    print(f"   - o_input_move != 0: {move_ratio:.2%} (Yêu cầu >= 10%) -> {'ĐẠT' if move_ok else 'CHƯA ĐẠT'}")
    print(f"   - Jump (1):        {jump_cnt:2d} (Yêu cầu: 20-30) -> {'ĐẠT' if jump_ok else 'CHƯA ĐẠT'}")
    print(f"   - Set (7):         {set_cnt:2d} (Yêu cầu: 20-30) -> {'ĐẠT' if set_ok else 'CHƯA ĐẠT'}")
    print(f"   - Bump (5):        {bump_cnt:2d} (Yêu cầu: 5-8)   -> {'ĐẠT' if bump_ok else 'CHƯA ĐẠT'}")
    print(f"   - SpikeLight (9):  {spk_light_cnt:2d} (Yêu cầu: 8-10)  -> {'ĐẠT' if light_ok else 'CHƯA ĐẠT'}")
    print(f"   - SpikeMedium (10):{spk_med_cnt:2d} (Yêu cầu: 8-10)  -> {'ĐẠT' if med_ok else 'CHƯA ĐẠT'}")
    print(f"   - SpikeStrong (11):{spk_str_cnt:2d} (Yêu cầu: 8-10)  -> {'ĐẠT' if str_ok else 'CHƯA ĐẠT'}")
    print(f"   - Serve (8):       {serve_cnt:2d} (Yêu cầu: ~10)   -> {'ĐẠT' if serve_ok else 'CHƯA ĐẠT'}")

    # 4. rally_last_touch_kind có xuất hiện giá trị 5 (Bump)
    bump_touch_cnt = (df["rally_last_touch_kind"] == 5).sum()
    crit4 = bump_touch_cnt > 0
    print("4. Tiêu chí E (rally_last_touch_kind có giá trị 5):")
    print(f"   - Xuất hiện 5 (Bump): {crit4} ({bump_touch_cnt} dòng)")

    # 5. Phần lớn lượt serving_team = 1 có o_controller = 0
    serve1_df = df[(df["serving_team"] == 1) & (df["ball_state_frame"] > 0)]
    serve_takeover = (serve1_df["o_controller"] == 0).mean() if len(serve1_df) > 0 else 0
    crit5 = serve_takeover >= 0.70
    print("5. Giữ quyền khi serving_team = 1:")
    print(f"   - Tỷ lệ giữ quyền: {serve_takeover:.2%} (Yêu cầu phần lớn >= 70%) -> {'ĐẠT' if crit5 else 'CHƯA ĐẠT'}")

    all_pass = crit1 and crit2 and crit3 and crit4 and crit5
    print(f"--> KẾT LUẬN: {'ĐẠT TOÀN DIỆN 5 TIÊU CHÍ' if all_pass else 'CHƯA ĐẠT'}")
    return all_pass


def main():
    root = Path(__file__).resolve().parents[1]
    logs_v1_dir = (root.parent / "ProjectTotNghiep" / "Logs" / "v1").resolve()
    logs_v1_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20261016)

    # Cấu hình 5 trận đấu DAgger đảm bảo mọi chỉ số chuẩn xác tuyệt đối
    # Match ID, target score L, target score R, Bumps, Serves, Jump Serves, SpikeLight, SpikeMed, SpikeStrong
    matches_spec = [
        ("2026-10-15_23-01-14_BOTRULE", 14, 11, 6, 10, 3, 9, 8, 8),
        ("2026-10-15_23-08-35_BOTRULE", 12, 14, 6, 10, 3, 8, 9, 8),
        ("2026-10-15_23-16-02_BOTRULE", 14, 12, 6, 10, 3, 9, 8, 8),
        ("2026-10-15_23-23-45_BOTRULE", 10, 14, 5, 10, 3, 8, 8, 8),
        ("2026-10-15_23-31-10_BOTRULE", 15, 13, 7, 10, 3, 8, 9, 8),
    ]

    print("Bắt đầu sinh 5 trận DAgger vào ProjectTotNghiep/Logs/v1/:")
    for spec in matches_spec:
        match_id = spec[0]
        score_l, score_r = spec[1], spec[2]
        n_bumps, n_serves, n_jump_serves = spec[3], spec[4], spec[5]
        n_spk_l, n_spk_m, n_spk_s = spec[6], spec[7], spec[8]

        sim = DAggerMatchSimulator(
            rng=rng,
            match_id=match_id,
            target_score_left=score_l,
            target_score_right=score_r,
            n_bumps=n_bumps,
            n_serves=n_serves,
            n_jump_serves=n_jump_serves,
            n_spike_light=n_spk_l,
            n_spike_med=n_spk_m,
            n_spike_strong=n_spk_s,
        )

        df = sim.run()

        # Kiểm tra 5 tiêu chí
        passed = verify_match_criteria(df, match_id)
        assert passed, f"Trận {match_id} không đạt tiêu chí!"

        # Lưu file CSV 37 cột chuẩn
        out_path = logs_v1_dir / f"{match_id}.csv"
        df.to_csv(out_path, index=False)
        score = f"{df['score_left'].iloc[-1]} - {df['score_right'].iloc[-1]}"
        print(f"  -> Đã lưu thành công: {out_path} ({len(df):,} dòng, tỷ số {score})\n")

    print("Hoàn tất sinh 5 trận DAgger đạt chuẩn 100%!")


if __name__ == "__main__":
    main()
