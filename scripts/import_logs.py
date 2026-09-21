"""Copy log CSV từ repo game sang data/raw/... của project AI.

    python scripts/import_logs.py --src ../ProjectTotNghiep/Logs --dest data/raw/v0

Bỏ qua file đã có (cùng tên), file rỗng và file chỉ có header.
"""

import argparse
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="thư mục Logs của game")
    parser.add_argument("--dest", default="data/raw/v0", help="thư mục đích trong project AI")
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = (ROOT / src).resolve()
    dest = ROOT / args.dest
    dest.mkdir(parents=True, exist_ok=True)

    copied = skipped = empty = 0
    for path in sorted(src.glob("*.csv")):
        with open(path, encoding="utf-8", errors="replace") as f:
            n_lines = sum(1 for _ in f)
        if n_lines <= 1:
            empty += 1
            continue
        target = dest / path.name
        if target.exists():
            skipped += 1
            continue
        shutil.copy2(path, target)
        copied += 1

    print(f"Copy {copied} file, bỏ qua {skipped} file đã có, {empty} file rỗng -> {dest}")


if __name__ == "__main__":
    main()
