"""Train các model theo một file config.

Cách dùng:
    python -m spike_ai.train --config configs/baseline_v0.yaml

Kết quả:
    models/<run_name>_<thời gian>/
        <target>__<model>.joblib   # model + metadata (feature, class, golden vectors)
        metrics.json               # toàn bộ chỉ số
        config.yaml                # bản sao config để tái lập
        cm_<target>__<model>.png   # confusion matrix
    reports/experiments.md         # thêm 1 dòng cho mỗi (target, model)
"""

import argparse
import json
import shutil
from datetime import datetime

import joblib
import yaml

from . import models as model_zoo
from .data import load_dirs
from .evaluate import compute_metrics, plot_confusion_matrix
from .features import build
from .paths import MODELS_DIR, REPORTS_DIR, resolve
from .schema import feature_names, load_spec
from .split import train_test_by_match

N_GOLDEN = 20  # số vector mẫu dùng để kiểm tra C++ ra cùng kết quả với Python


def load_config(path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    version, agent, seed = cfg["schema"], cfg["agent"], cfg.get("seed", 42)
    spec = load_spec(version)

    df = load_dirs(cfg["data_dirs"], expected_version=version)
    X, y = build(df, version, agent)
    groups = df["match_id"]
    train_idx, test_idx = train_test_by_match(groups, cfg.get("test_size", 0.2), seed)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = MODELS_DIR / f"{cfg['run_name']}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve(config_path), out_dir / "config.yaml")

    print(f"Dữ liệu: {df['match_id'].nunique()} trận, {len(df)} frame "
          f"(train {len(train_idx)} / test {len(test_idx)}) | schema {version} | agent {agent}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    all_metrics, rows = {}, []

    for target in cfg["targets"]:
        y_train, y_test = y[target].iloc[train_idx], y[target].iloc[test_idx]
        present = set(y_train) | set(y_test)
        labels = [v for v in spec["labels"][target]["values"] if v in present]

        for name, params in cfg["models"].items():
            model = model_zoo.create(name, seed, params)
            model.fit(X_train, y_train)
            m = compute_metrics(y_test, model.predict(X_test), labels)

            key = f"{target}__{name}"
            all_metrics[key] = m
            plot_confusion_matrix(m, key, out_dir / f"cm_{key}.png")

            golden = X_test.head(N_GOLDEN)
            joblib.dump(
                {
                    "model": model,
                    "model_name": name,
                    "target": target,
                    "agent": agent,
                    "schema": version,
                    "features": feature_names(spec),
                    "classes": [str(c) for c in model.classes_],
                    "golden_X": golden.to_numpy().tolist(),
                    "golden_proba": model.predict_proba(golden).tolist(),
                },
                out_dir / f"{key}.joblib",
            )
            rows.append((target, name, len(train_idx), len(test_idx), m["accuracy"], m["macro_f1"]))
            print(f"  {key:<32} acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    _append_experiment_log(out_dir.name, version, agent, rows)
    print(f"Đã lưu vào {out_dir}")
    return all_metrics


def _append_experiment_log(run_id, version, agent, rows) -> None:
    log = REPORTS_DIR / "experiments.md"
    today = datetime.now().strftime("%Y-%m-%d")
    with open(log, "a", encoding="utf-8") as f:
        for target, name, n_train, n_test, acc, f1 in rows:
            f.write(f"| {today} | {run_id} | {version} | {agent} | {target} | {name} "
                    f"| {n_train} | {n_test} | {acc:.3f} | {f1:.3f} | |\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="đường dẫn file YAML trong configs/")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
