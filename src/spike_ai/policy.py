"""Cách bot ML trong game RA QUYẾT ĐỊNH, chép sang Python để đo offline cho đúng.

Vì sao cần: `train.py` chấm điểm bằng argmax, còn game (`AIInputSystem.cpp`) không dùng argmax:
nó dùng ngưỡng P(ra tay), rút thăm có temperature, các luật chặn và luật nhảy sau Set. Model có
F1 argmax tốt chưa chắc cho ra con bot tốt, và ngược lại (đo được: argmax gần như luôn ra None
với model không rò rỉ nhãn, trong khi bot vẫn đánh được nhờ ngưỡng).

KHÔNG RÚT THĂM. Thay vào đó tính PHÂN PHỐI hành động của bot ở mỗi frame (bot sẽ làm gì với
xác suất bao nhiêu), rồi tính chỉ số kỳ vọng. Kết quả nhờ vậy không phụ thuộc seed và bằng đúng
trung bình của vô số lần rút thăm.

Thứ tự y hệt `AIInputSystem::update()` (repo game), mọi hằng số ở GAME_POLICY bên dưới:

  1. Có ra tay không: 1 - P(None) > act_threshold.
  2. Ra tay bằng gì: rút thăm theo P^(1/temperature), chỉ trong các cú dùng được
     (không None; không Jump nếu model_may_jump = false; Serve chỉ khi tới lượt mình giao).
  3. Phủ quyết SAU khi rút thăm: cú chạm bóng mà bóng ngoài vùng detect thì thành None.
     Ở game là `contactOutOfReach` - cú rút trúng bị bỏ, KHÔNG rút lại.
  4. Luật nhảy sau cú Set của mình: rút thăm một lần mỗi cú Set (p_jump_after_set); nhảy ở frame
     đầu tiên mà mô phỏng vật lý cho thấy sẽ gặp bóng sau [jump_min_t, jump_max_t] frame, và chỉ
     khi frame đó model không chọn một cú chạm.

CHƯA CHÉP (ghi rõ để đọc kết quả cho đúng):
  - Phần LUẬT GỐC của bot (di chuyển, PickAttackIntent) và luật "model không được xoá hành động
    của luật": cả hai cần quyết định của bot luật, mà dữ liệu offline là của người chơi.
  - Cú đập trên không sau khi bot tự nhảy: cần vị trí CỦA BOT đang nhảy, dữ liệu offline đi theo
    vị trí người chơi.
  - Lọc "cú đập phải rơi trong sân" (SpikeLandsInCourt): cần mô phỏng quỹ đạo cú đánh.
  - Cooldown: trong game, ra tay xong thì nhân vật bận vài frame nên không ra tay liên tiếp.
    Offline thì trạng thái đi theo người chơi, nên số lần ra tay kỳ vọng tính theo frame sẽ cao
    hơn trong game.
"""

import numpy as np
import pandas as pd

from .data import rebuild_kind_v1_to_v2

# Hằng số của game. PHẢI khớp Source/System/AIInputSystem.cpp - sửa bên đó thì sửa ở đây.
GAME_POLICY = {
    "act_threshold": 0.3,  # ACT_THRESHOLD
    "temperature": 3.0,  # PROBA_SAMPLING_TEMPERATURE
    "model_may_jump": False,  # MODEL_MAY_PROPOSE_JUMP
    "p_jump_after_set": 0.314,  # P_JUMP_AFTER_SET
    "jump_min_t": 18,  # JUMP_CONTACT_MIN_T
    "jump_max_t": 25,  # JUMP_CONTACT_MAX_T
    "jump_sim_frames": 70,  # JUMP_SIM_FRAMES
}

# Hình học của game, dùng cho vùng detect và mô phỏng cú nhảy.
PIXELS_PER_METER = 80.0
BALL_SIZE = 0.21 * PIXELS_PER_METER  # MatchObjectConfig::BallSize
CHAR_SIZE = {  # ComponentStorage.cpp: (rộng, cao)
    "player": (0.5 * PIXELS_PER_METER, 2.0 * PIXELS_PER_METER),
    "opponent": (0.5 * PIXELS_PER_METER, 1.5 * PIXELS_PER_METER),
}
WIDTH_PERCENT_CHANGE = 15.0  # SystemConfig
HEIGHT_PERCENT_CHANGE = -20.0
MIN_Y = 265.0  # SystemConfig::MIN_Y - mặt đất
GRAVITY = -9.81
JUMP_FRAME = 30
PARABOLIC = 1  # TrajectoryType::Parabolic

SPIKES = ("SpikeLight", "SpikeMedium", "SpikeStrong", "Spike")
CONTACTS = SPIKES + ("Set", "Bump", "Serve")
ACTION_STATE_SET = 6  # ActionState::Set (KHÁC FinalIntent::Set = 7)
TEAM = {"player": 0, "opponent": 1}  # Team::LEFT / RIGHT
ENTITY = {"player": 0, "opponent": 3}
RESET, ALIVE = 0, -1


def policy_params(cfg: dict | None) -> dict:
    """GAME_POLICY, ghi đè bằng mục `policy:` trong config nếu có (để thử ngưỡng khác)."""
    params = dict(GAME_POLICY)
    for key, value in (cfg or {}).get("policy", {}).items():
        if key not in params:
            raise ValueError(f"policy.{key} không tồn tại. Có: {sorted(params)}")
        params[key] = value
    return params


def _prefix(agent: str) -> str:
    return "p_" if agent == "player" else "o_"


def detect_percent(self_x, self_y, ball_x, ball_y, agent: str):
    """Khoảng cách tâm bóng - tâm nhân vật theo % chiều cao nhân vật (DetectPlayerBall trong game).

    Dùng toạ độ GAME (chưa lật sân): tâm nhân vật lệch +15% chiều rộng theo trục x thật, nên
    không lật được qua lưới.
    """
    w, h = CHAR_SIZE[agent]
    cx = np.asarray(self_x, float) + w * 0.5 + w * WIDTH_PERCENT_CHANGE / 100.0 / 2.0
    cy = np.asarray(self_y, float) + h * 0.5 + h * HEIGHT_PERCENT_CHANGE / 100.0 / 2.0
    dx = np.asarray(ball_x, float) + BALL_SIZE * 0.5 - cx
    dy = np.asarray(ball_y, float) + BALL_SIZE * 0.5 - cy
    return np.hypot(dx, dy) / h * 100.0


def in_detect_zone(percent, intent: str):
    """Ngưỡng y hệt AIInputSystem::BallInDetectZone: Spike* < 50, Bump <= 60, Set <= 100."""
    if intent in SPIKES:
        return percent < 50.0
    if intent == "Bump":
        return percent <= 60.0
    if intent == "Set":
        return percent <= 100.0
    return np.ones_like(percent, dtype=bool)


def frames_until_air_contact(rows: pd.DataFrame, agent: str, max_frames: int) -> np.ndarray:
    """Nếu nhảy ngay frame này, sau bao nhiêu frame bóng vào vùng đập (< 50%); -1 nếu không gặp.

    Chép AIInputSystem::FramesUntilAirContact: frame nhảy đầu +2*9,81 px, tới frame 30 mỗi frame
    +9,81 px, sau đó rơi 9,81 px/frame; bóng x += speed, y = a(x+b)^2 + c.
    """
    me = _prefix(agent)
    n = len(rows)
    out = np.full(n, -1, dtype=int)
    self_x = rows[f"{me}x"].to_numpy(float)
    self_y = rows[f"{me}y"].to_numpy(float).copy()
    bx = rows["ball_x"].to_numpy(float).copy()
    speed = rows["ball_speed"].to_numpy(float)
    a, b, c = (rows[k].to_numpy(float) for k in ("ball_a", "ball_b", "ball_c"))
    active = (rows["ball_traj_type"].to_numpy() == PARABOLIC).copy()
    rise = -2.0 * GRAVITY
    for t in range(1, max_frames + 1):
        if not active.any():
            break
        self_y += rise if t == 1 else (rise + GRAVITY if t <= JUMP_FRAME else GRAVITY)
        bx += speed
        by = a * (bx + b) * (bx + b) + c
        active &= (self_y > MIN_Y) & (by > MIN_Y)
        hit = active & (detect_percent(self_x, self_y, bx, by, agent) < 50.0)
        out[hit] = t
        active &= ~hit
    return out


def _after_own_set(rows: pd.DataFrame, agent: str) -> np.ndarray:
    """Cú chạm cuối là cú Set của chính mình (afterOwnSet trong game)."""
    kind = rows["rally_last_touch_kind"] if "rally_last_touch_kind" in rows else rows["_last_touch_kind"]
    return ((rows["rally_last_touch"] == ENTITY[agent]) & (kind == ACTION_STATE_SET)).to_numpy()


def add_last_touch_kind(df: pd.DataFrame) -> pd.DataFrame:
    """Gắn cột `_last_touch_kind` cho log v1 (suy ra như ADR-009), tính RIÊNG từng trận."""
    if "rally_last_touch_kind" in df:
        return df
    parts = [rebuild_kind_v1_to_v2(g) for _, g in df.groupby("match_id", sort=False)]
    return df.assign(_last_touch_kind=pd.concat(parts).reindex(df.index))


def action_distribution(rows: pd.DataFrame, proba: np.ndarray, classes, agent: str, params: dict) -> np.ndarray:
    """Phân phối hành động của bot ở mỗi frame, shape (n, len(classes)), mỗi hàng cộng lại = 1.

    `rows` là các dòng trạng thái lúc ra quyết định (đã ghép nhãn theo label_shift), cùng thứ tự
    với `proba` = model.predict_proba(X).
    """
    classes = [str(c) for c in classes]
    if "None" not in classes:
        raise ValueError("model phải có lớp None")
    k_none = classes.index("None")
    n = len(rows)
    me = _prefix(agent)

    serve_turn = ((rows["ball_state_frame"] == RESET) & (rows["serving_team"] == TEAM[agent])).to_numpy()
    usable = np.ones((n, len(classes)), dtype=bool)
    for k, name in enumerate(classes):
        if name == "None" or (name == "Jump" and not params["model_may_jump"]):
            usable[:, k] = False
        elif name == "Serve":
            usable[:, k] = serve_turn

    # 1-2. ngưỡng rồi rút thăm có temperature trong các cú dùng được
    weight = np.where(usable, np.power(np.clip(proba, 0.0, None), 1.0 / params["temperature"]), 0.0)
    total = weight.sum(axis=1, keepdims=True)
    acts = ((1.0 - proba[:, k_none]) > params["act_threshold"]) & (total[:, 0] > 0)
    dist = np.zeros_like(weight)
    dist[acts] = weight[acts] / total[acts]

    # 3. phủ quyết sau rút thăm: cú chạm ngoài vùng detect -> None
    percent = detect_percent(rows[f"{me}x"], rows[f"{me}y"], rows["ball_x"], rows["ball_y"], agent)
    for k, name in enumerate(classes):
        if name in CONTACTS:
            dist[~in_detect_zone(percent, name), k] = 0.0
    dist[:, k_none] = 1.0 - dist.sum(axis=1)

    # 4. luật nhảy sau cú Set của mình
    if "Jump" in classes and params["p_jump_after_set"] > 0:
        _apply_jump_rule(rows, dist, classes, agent, params)
    return dist


def _apply_jump_rule(rows, dist, classes, agent, params) -> None:
    k_jump, k_none = classes.index("Jump"), classes.index("None")
    me = _prefix(agent)
    after_set = _after_own_set(rows, agent)
    alive = (rows["ball_state_frame"] == ALIVE).to_numpy()
    on_ground = rows[f"{me}y"].to_numpy(float) <= MIN_Y
    candidate = after_set & alive & on_ground
    if not candidate.any():
        return

    t = np.full(len(rows), -1, dtype=int)
    idx = np.flatnonzero(candidate)
    t[idx] = frames_until_air_contact(rows.iloc[idx], agent, params["jump_sim_frames"])
    trigger = candidate & (t >= params["jump_min_t"]) & (t <= params["jump_max_t"])

    # Mỗi cú Set một lần nhảy: lấy frame kích hoạt ĐẦU TIÊN trong mỗi đoạn "sau cú Set của mình".
    match = rows["match_id"].to_numpy()
    new_segment = np.r_[True, (~after_set[:-1]) | (match[1:] != match[:-1])] & after_set
    segment = np.cumsum(new_segment)
    first = np.zeros(len(rows), dtype=bool)
    seen = set()
    for i in np.flatnonzero(trigger):
        if segment[i] not in seen:
            seen.add(segment[i])
            first[i] = True

    # Game chỉ nhảy khi frame đó model KHÔNG chọn cú chạm: phần xác suất "không chạm" chuyển sang Jump.
    contact = [k for k, name in enumerate(classes) if name in CONTACTS]
    no_contact = 1.0 - dist[:, contact].sum(axis=1)
    jump = params["p_jump_after_set"] * no_contact
    dist[first, k_jump] += jump[first]
    dist[first, k_none] -= jump[first]


# ---------------------------------------------------------------------------- chỉ số kỳ vọng
def expected_confusion(y_true, dist: np.ndarray, classes) -> np.ndarray:
    """C[i, j] = số frame kỳ vọng có nhãn thật classes[i] mà bot làm classes[j]."""
    classes = [str(c) for c in classes]
    y = np.asarray(y_true).astype(str)
    cm = np.zeros((len(classes), len(classes)))
    for i, name in enumerate(classes):
        mask = y == name
        if mask.any():
            cm[i] = dist[mask].sum(axis=0)
    return cm


def scores_from_confusion(cm: np.ndarray, classes, label_order) -> dict:
    """Precision / recall / F1 từng lớp và macro-F1 trên các lớp CÓ MẶT (như evaluate.compute_metrics)."""
    classes = [str(c) for c in classes]
    per_class, measured = {}, []
    for name in label_order:
        if name not in classes:
            continue
        k = classes.index(name)
        support = cm[k].sum()
        predicted = cm[:, k].sum()
        tp = cm[k, k]
        precision = tp / predicted if predicted > 0 else 0.0
        recall = tp / support if support > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(support),
            "predicted": float(predicted),
        }
        if support > 0:
            measured.append(f1)
    return {"macro_f1": float(np.mean(measured)) if measured else 0.0, "per_class": per_class}


def event_recall(y_true, dist: np.ndarray, classes, match_ids, tolerance: int) -> dict:
    """Với mỗi frame người chơi ra tay bằng cú c: xác suất bot cũng ra cú c trong +/- tolerance frame.

    Chấm F1 theo đúng frame quá khắt khe với hành động chỉ dài 1 frame: bot đập sớm 1 frame vẫn
    là đập đúng lúc. Số này trả lời "bot có làm đúng việc vào gần đúng lúc không".
    """
    classes = [str(c) for c in classes]
    y = np.asarray(y_true).astype(str)
    match_ids = np.asarray(match_ids)
    log_miss = np.log1p(-np.clip(dist, 0.0, 1.0 - 1e-12))
    result = {}
    for k, name in enumerate(classes):
        if name == "None":
            continue
        hits = []
        for m in pd.unique(match_ids):
            pos = np.flatnonzero(match_ids == m)
            targets = np.flatnonzero(y[pos] == name)
            if not len(targets):
                continue
            cum = np.r_[0.0, np.cumsum(log_miss[pos, k])]
            lo = np.clip(targets - tolerance, 0, len(pos))
            hi = np.clip(targets + tolerance + 1, 0, len(pos))
            hits.append(1.0 - np.exp(cum[hi] - cum[lo]))
        if hits:
            h = np.concatenate(hits)
            result[name] = {"recall": float(h.mean()), "support": int(len(h))}
    return result
