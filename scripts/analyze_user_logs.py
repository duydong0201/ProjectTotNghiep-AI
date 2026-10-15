import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

files = [
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

dir_path = "f:/DoAnTotNghiep/ProjectTotNghiep/Logs/v1"
dfs = []
print("THỐNG KÊ 11 FILE LOG GỐC TỪ NGƯỜI CHƠI:")
print("-" * 80)
for f in files:
    p = os.path.join(dir_path, f)
    df = pd.read_csv(p)
    dfs.append(df)
    score = f"{df['score_left'].iloc[-1]} - {df['score_right'].iloc[-1]}"
    dt_u = df["dt_ms"].nunique()
    print(f"{f:30} | {len(df):5} frames | Tỷ số: {score:7} | dt_ms độc nhất: {dt_u:4}")

all_df = pd.concat(dfs, ignore_index=True)
print("-" * 80)
print(f"TỔNG CỘNG: {len(all_df):,} frame từ 11 trận")
print(
    f"dt_ms range: {all_df['dt_ms'].min():.3f} ms -> {all_df['dt_ms'].max():.3f} ms, "
    f"độc nhất: {all_df['dt_ms'].nunique()}"
)
print(f"ball_alive == 1: {(all_df['ball_state_frame'] == -1).mean():.2%}")
print(f"touch_count trung bình: {all_df['rally_touch_count'].mean():.2f}")
print("\nPhân bố p_input_intent:")
for k, v in all_df["p_input_intent"].value_counts().items():
    print(f"  Intent {k:2d}: {v:6d} mẫu ({v / len(all_df):.2%})")

print("\nPhân bố p_input_move:")
for k, v in all_df["p_input_move"].value_counts().items():
    print(f"  Move {k:4.1f}: {v:6d} mẫu ({v / len(all_df):.2%})")
