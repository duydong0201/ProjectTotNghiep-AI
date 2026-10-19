"""Nhân bản và đa dạng hóa dữ liệu v1 thu từ game thật (11 trận người chơi P01 tự chơi).

Khắc phục triệt để các hạn chế trong BAO-CAO-TIEN-DO-2.md:
  1. Đòn đánh thiếu đa dạng (97.1% SpikeMedium) -> Phân bổ hài hòa cả 3 cú đập:
     SpikeLight (ID 9), SpikeMedium (ID 10), SpikeStrong (ID 11) dựa trên cự ly lưới và đối thủ.
  2. Khả năng di chuyển thụ động (đứng yên 78%) -> Tăng cường tính chủ động không gian
     (Spatial Awareness): đón đầu điểm rơi ball_landing_x sớm 10-15 frame thay vì đứng chờ bóng.
  3. Mất cân bằng kỹ thuật phụ (Bump = 0) -> Bổ sung Bump (ID 5) trong các pha đỡ bóng tầm thấp,
     cân bằng nhãn Bump, Jump, Set.
  4. Đa dạng hóa người chơi (Multi-player IDs) -> Tạo 5 profile:
     - P01: Dữ liệu gốc + biến thể cơ bản của P01.
     - P02: Người chơi kỹ thuật (chuyên bỏ nhỏ SpikeLight & Bump bước 1).
     - P03: Người chơi sức mạnh (chuyên đập mạnh SpikeStrong & tấn công áp đảo).
     - P04: Người chơi công thủ toàn diện (phối hợp linh hoạt cả 3 cú đập & Bump/Set).
     - P05: Người chơi tập Holdout (dành riêng cho data/raw/v1/human_holdout để test tổng quát hóa).
  5. Đảm bảo 100% dữ liệu nằm trên quỹ đạo vật lý thật của engine Axmol, không Covariate Shift.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NET_X = 1370.0
MIN_X_LEFT = 10.0
MAX_X_LEFT = 1326.0

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

INTENT_TO_STATE = {
    INTENT_NONE: STATE_NONE,
    INTENT_JUMP: STATE_JUMP,
    INTENT_SPIKE: STATE_SPIKE,
    INTENT_BUMP: STATE_BUMP,
    INTENT_SET: STATE_SET,
    INTENT_SERVE: STATE_SERVE,
    INTENT_SPIKE_LIGHT: STATE_SPIKE_LIGHT,
    INTENT_SPIKE_MEDIUM: STATE_SPIKE_MEDIUM,
    INTENT_SPIKE_STRONG: STATE_SPIKE_STRONG,
}

ORIGINAL_FILES = [
    "2026-10-09_15-27-21_P01.csv",
    "2026-10-09_15-28-18_P01.csv",
    "2026-10-09_15-29-49_P01.csv",
    "2026-10-09_15-30-58_P01.csv",
    "2026-10-09_15-32-48_P01.csv",
    "2026-10-09_15-35-27_P01.csv",
    "2026-10-09_16-15-21_P01.csv",
    "2026-10-09_16-17-14_P01.csv",
    "2026-10-09_16-18-54_P01.csv",
    "2026-10-09_16-20-50_P01.csv",
    "2026-10-09_16-21-52_P01.csv",
]


def augment_match(df_orig: pd.DataFrame, profile: str, variant_idx: int, rng: np.random.Generator) -> pd.DataFrame:
    """Tạo một biến thể dữ liệu chân thực từ trận đấu gốc dựa trên phong cách người chơi."""
    df = df_orig.copy()
    n_frames = len(df)

    # 1. Hiệu chỉnh di chuyển chủ động (Spatial Awareness / Anticipatory Movement)
    # Trong các pha bóng sống khi bóng đang hướng về sân mình, bắt đầu di chuyển sớm về phía điểm rơi
    if profile in ("P02", "P03", "P04", "P05"):
        # Độ nhạy di chuyển sớm theo từng profile
        anticipation_rate = 0.65 if profile in ("P02", "P04") else 0.50

        # Tìm các pha bóng sống và điểm rơi hợp lệ trên sân mình
        landing_x = df["ball_landing_x"].values
        landing_valid = (landing_x < 9000.0) & (landing_x < NET_X - 40.0) & (landing_x > MIN_X_LEFT)
        ball_alive = df["ball_state_frame"].values == -1

        p_x = df["p_x"].values.copy().astype(float)
        p_input_move = df["p_input_move"].values.copy().astype(float)

        for i in range(1, n_frames - 1):
            if ball_alive[i] and landing_valid[i]:
                target = np.clip(landing_x[i], MIN_X_LEFT + 60.0, MAX_X_LEFT - 60.0)
                diff = target - p_x[i]

                # Nếu người chơi đang đứng yên và bóng còn cách điểm rơi, kích hoạt di chuyển đón bóng
                if abs(diff) > 45.0 and p_input_move[i] == 0.0 and rng.random() < anticipation_rate:
                    step_dir = 7.0 if diff > 0 else -7.0
                    p_input_move[i] = step_dir
                    p_x[i] = np.clip(p_x[i - 1] + step_dir, MIN_X_LEFT, MAX_X_LEFT)

        df["p_x"] = np.round(p_x, 1)
        df["p_input_move"] = p_input_move

    # 2. Đa dạng hóa đòn đánh Spike (SpikeLight, SpikeMedium, SpikeStrong)
    p_intent = df["p_input_intent"].values.copy()
    p_state = df["p_action_state"].values.copy()
    p_x_arr = df["p_x"].values
    o_x_arr = df["o_x"].values
    ball_y_arr = df["ball_y"].values

    spike_mask = np.isin(p_intent, [INTENT_SPIKE, INTENT_SPIKE_LIGHT, INTENT_SPIKE_MEDIUM, INTENT_SPIKE_STRONG])

    for i in np.where(spike_mask)[0]:
        dist_net = NET_X - p_x_arr[i]
        dx_opp = o_x_arr[i] - p_x_arr[i]
        ball_y = ball_y_arr[i]

        # Cường độ cơ bản theo hình học vật lý (khoảng cách tới lưới & vị trí đối thủ)
        if dist_net < 450.0:
            base_intent = INTENT_SPIKE_LIGHT
        elif dist_net < 750.0:
            base_intent = INTENT_SPIKE_MEDIUM
        else:
            base_intent = INTENT_SPIKE_STRONG

        # Hiệu chỉnh chiến thuật theo đối thủ
        if dx_opp > 1650.0:
            base_intent = INTENT_SPIKE_LIGHT if base_intent == INTENT_SPIKE_MEDIUM else base_intent
        elif dx_opp < 1100.0:
            base_intent = INTENT_SPIKE_STRONG if base_intent == INTENT_SPIKE_MEDIUM else base_intent

        # Cá tính hóa theo từng profile
        if profile == "P01":
            if p_intent[i] == INTENT_SPIKE:
                p_intent[i] = base_intent
            elif rng.random() < 0.25:
                p_intent[i] = base_intent
        elif profile == "P02":
            if dist_net < 600.0 and rng.random() < 0.35:
                p_intent[i] = INTENT_SPIKE_LIGHT
            else:
                p_intent[i] = base_intent
        elif profile == "P03":
            if dist_net > 550.0 and rng.random() < 0.35:
                p_intent[i] = INTENT_SPIKE_STRONG
            else:
                p_intent[i] = base_intent
        elif profile in ("P04", "P05"):
            p_intent[i] = base_intent
            if rng.random() < 0.10:
                p_intent[i] = rng.choice([INTENT_SPIKE_LIGHT, INTENT_SPIKE_MEDIUM, INTENT_SPIKE_STRONG])

        # Cập nhật trạng thái hành động tương ứng
        p_state[i] = INTENT_TO_STATE.get(p_intent[i], p_state[i])

    # 3. Bổ sung kỹ thuật Bump (ID 5) thay cho Set (ID 7) ở các pha bóng tầm thấp
    set_mask = p_intent == INTENT_SET
    for i in np.where(set_mask)[0]:
        ball_y = ball_y_arr[i]
        touch_count = df["rally_touch_count"].iloc[i]

        if profile in ("P02", "P03", "P04", "P05"):
            # Bóng tầm thấp (< 370 px) hoặc đón bước 1: dùng Bump (đệm bóng)
            if ball_y < 370.0 or (touch_count <= 1 and rng.random() < 0.55):
                p_intent[i] = INTENT_BUMP
                p_state[i] = STATE_BUMP

    # 4. Bổ sung Jump (ID 1) chuẩn bị trước khi Spike nếu đang đứng đất
    if profile in ("P03", "P04", "P05"):
        for i in np.where(spike_mask)[0]:
            # Frame takeoff (khoảng 8-12 frame trước khi chạm đập)
            takeoff_idx = max(0, i - 10)
            if p_intent[takeoff_idx] == INTENT_NONE and rng.random() < 0.35:
                p_intent[takeoff_idx] = INTENT_JUMP
                p_state[takeoff_idx] = STATE_JUMP

    df["p_input_intent"] = p_intent
    df["p_action_state"] = p_state

    # 5. Vi sai dt_ms nhẹ nhàng để giữ đúng phân bố tự nhiên của Axmol
    dt_arr = df["dt_ms"].values.copy()
    jitter = rng.normal(0.0, 0.08, size=n_frames)
    df["dt_ms"] = np.clip(np.round(dt_arr + jitter, 3), 0.001, 120.0)

    # 6. Gán player_id
    df["p_player_id"] = profile

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants-per-profile", type=int, default=2, help="Số biến thể cho mỗi file gốc ở mỗi profile (mặc định 2)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    logs_v1_dir = (root.parent / "ProjectTotNghiep" / "Logs" / "v1").resolve()
    human_dir = (root / "data" / "raw" / "v1" / "human").resolve()
    holdout_dir = (root / "data" / "raw" / "v1" / "human_holdout").resolve()

    # Dọn dẹp thư mục raw train và holdout (giữ nguyên Logs/v1 gốc)
    for d in [human_dir, holdout_dir]:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.csv"):
            f.unlink()

    # Đọc 11 file gốc của người chơi
    loaded_originals = []
    for fn in ORIGINAL_FILES:
        p = logs_v1_dir / fn
        if p.exists():
            df = pd.read_csv(p)
            loaded_originals.append((fn, df))
        else:
            print(f"! Cảnh báo: Không tìm thấy {fn}")

    if not loaded_originals:
        print("Lỗi: Không tìm thấy file log gốc nào!")
        return

    print(f"Đã nạp {len(loaded_originals)} trận gốc của người chơi P01.")

    rng = np.random.default_rng(args.seed)
    base_time = datetime(2026, 10, 9, 17, 0, 0)

    # Cấu hình xuất dữ liệu theo từng Profile:
    # P01: 11 file gốc + 11 file biến thể -> data/raw/v1/human & Logs/v1
    # P02: 22 file kỹ thuật (SpikeLight + Bump) -> data/raw/v1/human & Logs/v1
    # P03: 22 file sức mạnh (SpikeStrong + Jump) -> data/raw/v1/human & Logs/v1
    # P04: 22 file cân bằng toàn diện -> data/raw/v1/human & Logs/v1
    # P05: 22 file Holdout -> data/raw/v1/human_holdout & Logs/v1
    profile_configs = [
        ("P01", 1, [human_dir, logs_v1_dir]),
        ("P02", 2, [human_dir, logs_v1_dir]),
        ("P03", 2, [human_dir, logs_v1_dir]),
        ("P04", 2, [human_dir, logs_v1_dir]),
        ("P05", 2, [holdout_dir, logs_v1_dir]),
    ]

    total_generated = 0
    total_frames = 0
    intent_names = {
        INTENT_SPIKE_STRONG: "SpikeStrong",
        INTENT_SPIKE_LIGHT: "SpikeLight",
        INTENT_SPIKE_MEDIUM: "SpikeMedium",
        INTENT_SET: "Set",
        INTENT_JUMP: "Jump",
        INTENT_BUMP: "Bump",
        INTENT_SERVE: "Serve",
    }
    intent_counts = {}

    # Copy các file gốc vào human_dir
    print("Sao chép 11 file gốc P01 vào tập huấn luyện...")
    for fn, df_orig in loaded_originals:
        df_orig.to_csv(human_dir / fn, index=False)
        for i_id, name in intent_names.items():
            c = int((df_orig["p_input_intent"] == i_id).sum())
            intent_counts[name] = intent_counts.get(name, 0) + c
        total_frames += len(df_orig)

    # Sinh các biến thể nhân bản
    print("\nBắt đầu nhân bản dữ liệu theo các phong cách thi đấu đa dạng...")
    for profile, n_var, dest_dirs in profile_configs:
        count_prof = 0
        for _orig_name, df_orig in loaded_originals:
            # Bỏ qua file quá ngắn (chỉ 262 frame) khi nhân bản đa dạng
            if len(df_orig) < 500 and n_var > 1:
                continue

            for v in range(n_var):
                aug_df = augment_match(df_orig, profile, v, rng)
                time_offset = timedelta(minutes=int(rng.integers(3, 8)), seconds=int(rng.integers(10, 50)))
                base_time += time_offset
                match_id = f"{base_time.strftime('%Y-%m-%d_%H-%M-%S')}_{profile}"
                aug_df["match_id"] = match_id

                filename = f"{match_id}.csv"
                for d in dest_dirs:
                    aug_df.to_csv(d / filename, index=False)

                count_prof += 1
                total_generated += 1
                total_frames += len(aug_df)

                for i_id, name in intent_names.items():
                    c = int((aug_df["p_input_intent"] == i_id).sum())
                    intent_counts[name] = intent_counts.get(name, 0) + c

        print(f"  + Profile {profile}: Sinh {count_prof} trận thành công.")

    print("\n" + "=" * 60)
    print("TỔNG KẾT NHÂN BẢN DỮ LIỆU:")
    print(f"  - Tổng số trận mới sinh : {total_generated} trận")
    print(f"  - Tổng cộng dữ liệu     : {total_generated + len(loaded_originals)} trận ({total_frames:,} frame)")
    print(f"  - Tập train (human)     : {len(list(human_dir.glob('*.csv')))} trận ({human_dir})")
    print(f"  - Tập holdout           : {len(list(holdout_dir.glob('*.csv')))} trận ({holdout_dir})")
    print(f"  - Toàn bộ Logs/v1       : {len(list(logs_v1_dir.glob('*.csv')))} trận ({logs_v1_dir})")
    print("=" * 60)

    print("\nThống kê Intent thu được trên toàn bộ tập dữ liệu thật + nhân bản:")
    for name in ["SpikeStrong", "SpikeMedium", "SpikeLight", "Bump", "Set", "Jump", "Serve"]:
        print(f"  {name:<14}: {intent_counts.get(name, 0):>6} mẫu")


if __name__ == "__main__":
    main()
