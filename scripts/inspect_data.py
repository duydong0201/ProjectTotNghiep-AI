"""Xem nhanh dữ liệu trước khi train.

python scripts/inspect_data.py --config configs/baseline_v0.yaml
"""

import argparse

import yaml

from spike_ai.data import load_dirs
from spike_ai.features import build
from spike_ai.paths import resolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(resolve(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_dirs(cfg["data_dirs"], expected_version=cfg["schema"])
    X, y = build(df, cfg["schema"], cfg["agent"])

    frames = df.groupby("match_id").size()
    print(f"Schema {cfg['schema']} | agent {cfg['agent']}")
    print(
        f"Số trận: {len(frames)} | tổng frame: {len(df)} | "
        f"frame/trận: min {frames.min()}, trung bình {frames.mean():.0f}, max {frames.max()}"
    )

    for target in cfg["targets"]:
        counts = y[target].value_counts()
        print(f"\nPhân bố nhãn '{target}':")
        for label, n in counts.items():
            print(f"  {str(label):<12} {n:>7}  ({n / len(y):6.2%})")

    print("\nThống kê feature:")
    print(X.describe().T[["mean", "std", "min", "max"]].round(2).to_string())


if __name__ == "__main__":
    main()
