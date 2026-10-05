"""Sinh log v0 TỔNG HỢP để kiểm chứng pipeline và dựng learning curve.

    python scripts/make_synthetic_v0.py --matches 200
    python scripts/make_synthetic_v0.py --matches 200 --out data/raw/v0_synthetic --seed 42

ĐÂY KHÔNG PHẢI DỮ LIỆU NGƯỜI CHƠI. Nhãn do hàm `spike_policy()` dưới đây sinh ra, không
phải do ai bấm phím. Ba hệ quả bắt buộc phải hiểu trước khi dùng:

1. Model train trên dữ liệu này chỉ học lại `spike_policy()`. macro-F1 cao KHÔNG chứng minh
   điều gì về hành vi người chơi - nó chỉ chứng minh pipeline hoạt động.
2. Không bao giờ ghi vào `Logs/` của game hay `data/raw/v0` (dữ liệu người chơi thật). Mặc
   định ghi ra `data/raw/v0_synthetic/`, tên file có tiền tố `synth_` để không lẫn.
3. Trong báo cáo phải gọi đúng tên: "dữ liệu tổng hợp theo quy luật đã biết", và dẫn chính
   file này làm định nghĩa quy luật.

DÙNG ĐƯỢC CHO:
  - Kiểm chứng pipeline ở quy mô lớn: code train/export/golden test có chịu được N lớn không.
  - Sanity check: pipeline có khôi phục lại được một quy luật ĐÃ BIẾT không? Nếu không thì
    có bug ở feature hoặc nhãn, phát hiện được mà không cần chờ dữ liệu thật.
  - Learning curve: macro-F1 thay đổi thế nào theo số mẫu. Đây là bằng chứng định lượng cho
    kết luận "cần thu thêm dữ liệu", xem scripts/learning_curve.py.

KHÔNG DÙNG ĐƯỢC CHO:
  - Bất kỳ phát biểu nào về việc bot bắt chước người chơi.
  - Model đưa vào game. Nó sẽ chơi theo quy luật bịa, không theo người.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spike_ai.paths import resolve

# Hằng số lấy đúng từ game (Source/Config/System/SystemConf.h, Source/Config/Match/BigRect.h)
NET_X = 1370.0
PIXELS_PER_METER = 80.0
LEFT_9M_LINE_X = NET_X - 9.0 * PIXELS_PER_METER  # 650
MIN_X_LEFT, MAX_X_LEFT = 10.0, 1326.0
MIN_X_RIGHT, MAX_X_RIGHT = 1346.0, 2720.0
MIN_Y = 265.0
SPEED = 7.0

SPIKES = ("SpikeLight", "SpikeMedium", "SpikeStrong")

# Tỉ lệ nhãn bị đổi sang cú khác. Người chơi không tất định; không có nhiễu thì model đạt
# macro-F1 ~1.0 và sanity check mất ý nghĩa vì không phân biệt được "pipeline đúng" với
# "bài toán quá dễ".
LABEL_NOISE = 0.12


def spike_policy(self_x: float, opp_x: float, rng: np.random.Generator) -> str:
    """QUY LUẬT SINH NHÃN - đây là thứ model sẽ học lại.

    Dựa trên khoảng cách tới lưới, cùng ý tưởng vật lý với `PickAttackIntent` của game: đứng
    càng xa lưới thì càng phải đánh mạnh để bóng sang được sân đối phương; đứng sát lưới mà
    đánh mạnh thì bóng bay ra ngoài biên.

    Có thêm một hiệu chỉnh theo vị trí đối thủ: đối thủ đứng xa thì đánh mạnh hơn một bậc
    (đủ chỗ để bóng rơi), đối thủ đứng gần thì nhẹ hơn một bậc. Mục đích là để nhãn phụ
    thuộc NHIỀU HƠN MỘT feature, nếu không thì bài toán chỉ là một ngưỡng cắt trên self_x.
    """
    # Ngưỡng chia sân thành 3 dải xấp xỉ bằng nhau (nhân vật đi lại đều trong 10..1326, nên
    # dist_to_net rải trong 44..1360). Chia đều để ba lớp cân nhau - dữ liệu sanity check mà
    # lệch 74% về một lớp thì khó đọc kết quả.
    dist_to_net = NET_X - self_x

    if dist_to_net < 450.0:
        base = 0
    elif dist_to_net < 900.0:
        base = 1
    else:
        base = 2

    # Ngưỡng theo phân bố thật của dx_opp (trung bình ~1365, khoảng 20..2710)
    dx_opp = opp_x - self_x
    if dx_opp > 1700.0:
        base += 1
    elif dx_opp < 1050.0:
        base -= 1

    base = int(np.clip(base, 0, 2))
    if rng.random() < LABEL_NOISE:
        base = int(rng.integers(0, 3))
    return SPIKES[base]


def _walk(rng: np.random.Generator, n: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Quỹ đạo đi lại trong [lo, hi] theo bước SPEED. Trả về (vị trí, nhãn hướng: -1/0/+1)."""
    x = np.empty(n)
    direction = np.zeros(n, dtype=int)
    pos = rng.uniform(lo, hi)
    target = rng.uniform(lo, hi)
    for i in range(n):
        if abs(target - pos) <= SPEED:
            target = rng.uniform(lo, hi)  # tới đích thì chọn đích mới
        step = SPEED if target > pos else -SPEED
        if rng.random() < 0.08:
            step = 0.0  # thỉnh thoảng đứng lại, để nhãn có lớp "đứng yên"
        pos = float(np.clip(pos + step, lo, hi))
        x[i] = pos
        direction[i] = 0 if step == 0.0 else (1 if step > 0 else -1)
    return x, direction


def make_match(rng: np.random.Generator, n_frames: int) -> pd.DataFrame:
    px, pdir = _walk(rng, n_frames, MIN_X_LEFT, MAX_X_LEFT)
    bx, bdir = _walk(rng, n_frames, MIN_X_RIGHT, MAX_X_RIGHT)

    p_event = np.where(pdir > 0, "MoveRight", np.where(pdir < 0, "MoveLeft", "None")).astype(object)
    b_event = np.where(bdir > 0, "MoveRight", np.where(bdir < 0, "MoveLeft", "None")).astype(object)

    # Các frame chạm bóng, cách nhau 40..140 frame như một pha bóng thật
    frame = 0
    while True:
        frame += int(rng.integers(40, 140))
        if frame >= n_frames:
            break
        # Người chơi: chủ yếu đập, đôi khi đỡ / nâng / giao bóng (giống phân bố log thật)
        roll = rng.random()
        if roll < 0.55:
            p_event[frame] = spike_policy(px[frame], bx[frame], rng)
        elif roll < 0.80:
            p_event[frame] = "Bump"
        elif roll < 0.90:
            p_event[frame] = "Set"
        else:
            p_event[frame] = "Serve"
        # Bot bên phải: lật toạ độ về góc nhìn của nó trước khi áp dụng cùng quy luật
        if rng.random() < 0.6:
            b_frame = min(frame + int(rng.integers(10, 40)), n_frames - 1)
            b_event[b_frame] = spike_policy(2.0 * NET_X - bx[b_frame], 2.0 * NET_X - px[b_frame], rng)

    return pd.DataFrame(
        {
            "Frame": np.arange(1, n_frames + 1),
            "PlayerPosX": px.round(),
            "PlayerPosY": MIN_Y,
            "PlayerEvent": p_event,
            "BotPosX": bx.round(),
            "BotPosY": MIN_Y,
            "BotEvent": b_event,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", type=int, default=200, help="số trận cần sinh")
    parser.add_argument("--out", default="data/raw/v0_synthetic", help="thư mục đích")
    parser.add_argument("--seed", type=int, default=42, help="seed để tái lập đúng bộ dữ liệu")
    args = parser.parse_args()

    out = resolve(args.out)
    if out.name in {"v0", "v1", "Logs"} or "Logs" in out.parts:
        sys.exit(f"TỪ CHỐI: {out} là nơi chứa dữ liệu thật. Dùng một thư mục riêng.")
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    total = 0
    for i in range(args.matches):
        df = make_match(rng, int(rng.integers(1500, 5000)))
        df.to_csv(out / f"synth_{i:04d}.csv", index=False)
        total += len(df)

    print(f"Đã sinh {args.matches} trận / {total:,} frame vào {out}")
    print(f"seed={args.seed}, nhiễu nhãn={LABEL_NOISE:.0%}, quy luật = spike_policy() trong {Path(__file__).name}")
    print("ĐÂY LÀ DỮ LIỆU TỔNG HỢP, không phải hành vi người chơi. Xem docstring đầu file.")


if __name__ == "__main__":
    main()
