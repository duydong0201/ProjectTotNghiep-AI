"""Train các model theo một file config.

Cách dùng:
    python -m spike_ai.train --config configs/baseline_v0.yaml           # train, đo trên dev
    python -m spike_ai.train --config configs/baseline_v0.yaml --final   # đo thêm trên holdout

Chỉ dùng --final khi đã CHỐT model (hết GĐ4, trước khi viết báo cáo). Kết quả holdout
chỉ để ghi nhận, không được dùng để sửa feature/model (xem docs/workflow.md).

Kết quả:
    models/<run_name>_<thời gian>/
        <target>__<model>.joblib   # model + metadata (feature, class, golden vectors)
        metrics.json               # toàn bộ chỉ số
        split.json                 # nhóm nào nằm ở train / dev / holdout
        config.yaml                # bản sao config để tái lập
        cm_<eval>_<target>__<model>.png   # confusion matrix
    reports/experiments.md         # thêm 1 dòng cho mỗi (tập đo, target, model)
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
from .split import DEV, HOLDOUT, TRAIN, group_keys, split_by_hash, split_manifest

N_GOLDEN = 20  # số vector mẫu dùng để kiểm tra C++ ra cùng kết quả với Python


def load_config(path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str, final: bool = False) -> dict:
    cfg = load_config(config_path)
    version, agent, seed = cfg["schema"], cfg["agent"], cfg.get("seed", 42)
    split_cfg = cfg["split"]
    spec = load_spec(version)

    df = load_dirs(cfg["data_dirs"], expected_version=version)
    X, y = build(df, version, agent)
    groups = group_keys(df, split_cfg.get("group_by", "match"), agent)
    index = split_by_hash(groups, split_cfg["salt"], split_cfg["dev_ratio"], split_cfg["holdout_ratio"])
    eval_sets = [DEV, HOLDOUT] if final else [DEV]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = MODELS_DIR / f"{cfg['run_name']}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve(config_path), out_dir / "config.yaml")
    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest(groups, index), f, indent=2, ensure_ascii=False)

    counts = " / ".join(f"{name} {groups.iloc[idx].nunique()} nhóm, {len(idx)} frame" for name, idx in index.items())
    print(f"Schema {version} | agent {agent} | {counts}")
    if not final:
        print("  (holdout được giữ kín - chỉ đo khi chạy với --final)")

    X_train = X.iloc[index[TRAIN]]
    all_metrics, rows = {}, []

    for target in cfg["targets"]:
        y_train = y[target].iloc[index[TRAIN]]

        for name, params in cfg["models"].items():
            model = model_zoo.create(name, seed, params)
            model.fit(X_train, y_train)
            key = f"{target}__{name}"

            for eval_name in eval_sets:
                X_eval, y_eval = X.iloc[index[eval_name]], y[target].iloc[index[eval_name]]
                present = set(y_train) | set(y_eval)
                labels = [v for v in spec["labels"][target]["values"] if v in present]
                m = compute_metrics(y_eval, model.predict(X_eval), labels)

                all_metrics[f"{eval_name}/{key}"] = m
                plot_confusion_matrix(m, f"{eval_name}: {key}", out_dir / f"cm_{eval_name}_{key}.png")
                rows.append((eval_name, target, name, len(X_train), len(X_eval), m["accuracy"], m["macro_f1"]))
                print(f"  [{eval_name:<7}] {key:<32} acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}")

            # golden vectors lấy từ dev: dữ liệu model chưa thấy, nhưng không phải holdout
            golden = X.iloc[index[DEV]].head(N_GOLDEN)
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

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    _append_experiment_log(out_dir.name, version, agent, rows)
    print(f"Đã lưu vào {out_dir}")
    return all_metrics


def _append_experiment_log(run_id, version, agent, rows) -> None:
    log = REPORTS_DIR / "experiments.md"
    today = datetime.now().strftime("%Y-%m-%d")
    with open(log, "a", encoding="utf-8") as f:
        for eval_name, target, name, n_train, n_eval, acc, f1 in rows:
            f.write(
                f"| {today} | {run_id} | {version} | {agent} | {target} | {name} "
                f"| {eval_name} | {n_train} | {n_eval} | {acc:.3f} | {f1:.3f} | |\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="đường dẫn file YAML trong configs/")
    parser.add_argument("--final", action="store_true", help="đo thêm trên holdout (chỉ khi đã chốt model)")
    args = parser.parse_args()
    run(args.config, final=args.final)


if __name__ == "__main__":
    main()
