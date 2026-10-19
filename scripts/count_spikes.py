"""Đếm số mẫu 3 cú đập trong log, để biết đã thu đủ dữ liệu train `spike_choice` chưa.

    python scripts/count_spikes.py                        # đếm data/raw/v0
    python scripts/count_spikes.py --dirs data/raw/v1     # đếm thư mục khác
    python scripts/count_spikes.py --target 200           # đổi mục tiêu mỗi lớp

Đếm riêng cho nhân vật sân trái (agent=player) và sân phải (agent=opponent).

CHÚ Ý về nguồn gốc dữ liệu: log không ghi ai điều khiển nhân vật nào (schema v0 thiếu cột
`controller`). Nhân vật trái CHỈ là người thật khi log thu ở chế độ người-đánh-bot. Nếu game
bật `ENABLE_BOT_VS_BOT` thì cả hai bên đều là bot - loại log đó phải để thư mục riêng
(`data/raw/v0_botvsbot`) và không dùng cho mục tiêu "bot chơi giống người".
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Cho chay bang `python scripts/count_spikes.py` ma khong can PYTHONPATH hay `pip install -e .`.
# Phai dat truoc cac import cua spike_ai, nen hai dong duoi vi pham E402 mot cach co y.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spike_ai.data import load_dirs  # noqa: E402
from spike_ai.features import SPIKE_CHOICES, build  # noqa: E402

# Dưới ngưỡng này thì F1 của lớp đó quá nhiễu để kết luận (xem evaluate.MIN_SUPPORT = 10,
# nhưng để train được tử tế cần nhiều hơn hẳn).
DEFAULT_TARGET = 150

AGENT_LABEL = {"player": "Nhân vật SÂN TRÁI (PLAYER)", "opponent": "Nhân vật SÂN PHẢI (OPPONENT_1)"}


def count(dirs: list[str], per_class_target: int) -> bool:
    df = load_dirs(dirs)
    print(f"Đọc {len(df):,} frame / {df['match_id'].nunique()} trận từ {dirs}\n")

    enough_on_left = False
    for agent, who in AGENT_LABEL.items():
        _, y = build(df, _version_of(df), agent)
        counts = y["spike_choice"].value_counts()
        total = int(counts.sum())

        print(f"--- {who} ---")
        for name in SPIKE_CHOICES:
            n = int(counts.get(name, 0))
            missing = max(0, per_class_target - n)
            mark = "OK " if missing == 0 else "còn thiếu"
            print(f"  {name:<12} {n:>5}  {mark} {missing if missing else ''}")
        print(f"  {'tổng':<12} {total:>5}")

        if agent == "player":
            enough_on_left = all(int(counts.get(n, 0)) >= per_class_target for n in SPIKE_CHOICES)
        print()

    if not _has_ball(df):
        print("! Log này là schema v0 - KHÔNG có vị trí bóng, nên dù đủ mẫu vẫn không train được")
        print("  spike_choice: quyết định đập cú nào phụ thuộc vị trí/quỹ đạo bóng.")
        print("  Cần log v1 theo schema/feature_spec.v1.json (xem docs/data_contract.md).")
        return False

    if enough_on_left:
        print(f"Sân trái đã đủ {per_class_target}+ mẫu mỗi lớp -> train được spike_choice.")
        print("Nhớ kiểm tra log này thu ở chế độ nào: bot-vs-bot thì sân trái cũng là bot.")
    else:
        print("Sân trái chưa đủ mẫu. Cách nhanh nhất: chơi vài trận và cố ý đập bóng mỗi pha.")
    return enough_on_left


def _version_of(df) -> str:
    return "v1" if _has_ball(df) else "v0"


def _has_ball(df) -> bool:
    return "ball_x" in df.columns


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", default=["data/raw/v0"], help="thư mục log cần đếm")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="số mẫu mong muốn cho mỗi cú đập")
    args = parser.parse_args()
    sys.exit(0 if count(args.dirs, args.target) else 1)


if __name__ == "__main__":
    main()
