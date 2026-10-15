import glob
import os
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logs_dir = "../ProjectTotNghiep/Logs/v1"
human_dir = "data/raw/v1/human"
holdout_dir = "data/raw/v1/human_holdout"
real_dir = "data/raw/v1/human_real"

os.makedirs(real_dir, exist_ok=True)
os.makedirs(human_dir, exist_ok=True)
os.makedirs(holdout_dir, exist_ok=True)

real_files = [
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
    "2026-10-09_17-23-29_P01.csv",
]

# 1. Lưu trữ 12 file thật vào human_real
for rf in real_files:
    src = os.path.join(logs_dir, rf)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(real_dir, rf))
print(f"Đã lưu trữ {len(real_files)} file thật độc nhất vào: {real_dir}")

# 2. Xóa các file bản sao sinh ra trong Logs/v1
deleted = 0
for f in glob.glob(os.path.join(logs_dir, "*.csv")):
    name = os.path.basename(f)
    if not name.startswith("2026-10-06") and name not in real_files:
        os.remove(f)
        deleted += 1
print(f"Đã xóa {deleted} file bản sao/nhân bản khỏi: {logs_dir}")

# 3. Khôi phục data/raw/v1/human (120 file P01, P02, P03) và holdout (30 file P04) từ 2026-10-06
for f in glob.glob(os.path.join(human_dir, "*.csv")):
    os.remove(f)
for f in glob.glob(os.path.join(holdout_dir, "*.csv")):
    os.remove(f)

for f in sorted(glob.glob(os.path.join(logs_dir, "2026-10-06*.csv"))):
    name = os.path.basename(f)
    if "_P04.csv" in name:
        shutil.copy2(f, os.path.join(holdout_dir, name))
    else:
        shutil.copy2(f, os.path.join(human_dir, name))

n_human = len(glob.glob(os.path.join(human_dir, "*.csv")))
n_holdout = len(glob.glob(os.path.join(holdout_dir, "*.csv")))
n_logs = len(glob.glob(os.path.join(logs_dir, "*.csv")))
n_real = len(glob.glob(os.path.join(real_dir, "*.csv")))

print("Khôi phục thành công:")
print(f"  - data/raw/v1/human         : {n_human} trận (120 trận sạch 2026-10-06: P01, P02, P03)")
print(f"  - data/raw/v1/human_holdout : {n_holdout} trận (30 trận sạch 2026-10-06: P04)")
print(f"  - data/raw/v1/human_real    : {n_real} trận (12 trận người chơi thật 2026-10-09)")
print(f"  - Logs/v1                   : {n_logs} trận (150 trận 2026-10-06 + 12 trận người thật)")
