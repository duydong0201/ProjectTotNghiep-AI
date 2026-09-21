"""Train các model theo một file config.

Cách dùng:
    python -m spike_ai.train --config configs/baseline_v0.yaml           # train, đánh giá trên dev / CV
    python -m spike_ai.train --config configs/human_v1.yaml --final      # đo thêm trên holdout

Chế độ đánh giá (split.dev_mode trong config), xem ADR-004:
    single  fit trên train, đo trên một tập dev (chia bằng hash). Hợp khi có nhiều trận.
    kfold   cross-validation theo nhóm trên toàn bộ dữ liệu không thuộc holdout.
            Báo cáo trung bình ± độ lệch chuẩn. Hợp khi ít trận.
    lopo    leave-one-player-out: mỗi fold bỏ ra một người chơi. Trả lời câu hỏi
            "model có tổng quát sang người chơi mới không?".

Holdout (chỉ đo khi có --final, và chỉ khi đã CHỐT model):
    split.holdout_ratio > 0   holdout = các nhóm được hash chọn ra từ data_dirs
    holdout_dirs: [...]       holdout "tương lai" = dữ liệu thu SAU khi chốt model, để ở thư mục riêng

Kết quả:
    models/<run_name>_<thời gian>/
        <target>__<model>.joblib   # model + metadata (feature, class, golden vectors)
        metrics.json               # toàn bộ chỉ số
        split.json                 # nhóm nào ở tập nào + báo cáo cân bằng + cảnh báo
        config.yaml                # bản sao config để tái lập
        cm_<tập đo>_<target>__<model>.png
    reports/experiments.md         # thêm 1 dòng cho mỗi (tập đo, target, model)
"""

import argparse
import json
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yaml

from . import models as model_zoo
from .data import load_dirs
from .evaluate import compute_metrics, plot_confusion_matrix, summarize_folds
from .features import build
from .paths import MODELS_DIR, REPORTS_DIR, resolve
from .schema import feature_names, load_spec
from .split import (
    DEV,
    HOLDOUT,
    TRAIN,
    assert_no_overlap,
    cv_folds,
    group_keys,
    split_by_hash,
    split_manifest,
    split_report,
)

N_GOLDEN = 20  # số vector mẫu dùng để kiểm tra C++ ra cùng kết quả với Python
DEV_MODES = ("single", "kfold", "lopo")


def load_config(path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(dirs, version: str, game_versions: list[str] | None) -> pd.DataFrame:
    """Đọc log; nếu config có game_versions thì chỉ giữ log từ các phiên bản game đó (schema v1)."""
    df = load_dirs(dirs, expected_version=version)
    if game_versions:
        if "game_version" not in df.columns:
            raise ValueError("game_versions chỉ dùng được với log có cột game_version (schema v1)")
        df = df[df["game_version"].astype(str).isin([str(v) for v in game_versions])].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Không còn dòng nào sau khi lọc game_versions={game_versions}")
    return df


def validate_split_config(cfg: dict) -> dict:
    s = dict(cfg["split"])
    s.setdefault("group_by", "match")
    s.setdefault("dev_mode", "single")
    s.setdefault("dev_ratio", 0.0)
    s.setdefault("holdout_ratio", 0.0)
    s.setdefault("n_folds", 5)
    if s["dev_mode"] not in DEV_MODES:
        raise ValueError(f"split.dev_mode phải là một trong {DEV_MODES}")
    if s["dev_mode"] == "single" and s["dev_ratio"] <= 0:
        raise ValueError("dev_mode=single cần dev_ratio > 0")
    if s["dev_mode"] != "single" and s["dev_ratio"] > 0:
        raise ValueError(f"dev_mode={s['dev_mode']} dùng cross-validation, hãy đặt dev_ratio: 0")
    if cfg.get("holdout_dirs") and s["holdout_ratio"] > 0:
        raise ValueError("Chỉ chọn một loại holdout: holdout_ratio > 0 HOẶC holdout_dirs")
    return s


def run(config_path: str, final: bool = False) -> dict:
    cfg = load_config(config_path)
    version, agent, seed = cfg["schema"], cfg["agent"], cfg.get("seed", 42)
    s = validate_split_config(cfg)
    spec = load_spec(version)

    df = load_data(cfg["data_dirs"], version, cfg.get("game_versions"))
    X, y = build(df, version, agent)
    groups = group_keys(df, s["group_by"], agent)
    index = split_by_hash(groups, s["salt"], s["dev_ratio"], s["holdout_ratio"])

    # ---- holdout: chỉ nạp / đụng tới khi --final
    holdout = None
    if final:
        holdout = _load_holdout(cfg, s, df, X, y, index, version, agent)

    # ---- báo cáo cân bằng (không mô tả holdout nếu chưa --final)
    visible = {k: v for k, v in index.items() if k != HOLDOUT}
    report, warnings = split_report(groups, visible, {t: y[t] for t in cfg["targets"]})

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = MODELS_DIR / f"{cfg['run_name']}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(resolve(config_path), out_dir / "config.yaml")

    has_holdout = bool(len(index[HOLDOUT]) or cfg.get("holdout_dirs"))
    _print_split(version, agent, s, report, warnings, holdout, has_holdout)
    all_metrics, rows = {}, []

    for target in cfg["targets"]:
        order = spec["labels"][target]["values"]
        for name, params in cfg["models"].items():
            key = f"{target}__{name}"

            if s["dev_mode"] == "single":
                fit_idx = index[TRAIN]
                model = model_zoo.create(name, seed, params).fit(X.iloc[fit_idx], y[target].iloc[fit_idx])
                dev_idx = index[DEV]
                m = compute_metrics(y[target].iloc[dev_idx], model.predict(X.iloc[dev_idx]), order)
                eval_name, golden = "dev", X.iloc[dev_idx].head(N_GOLDEN)
                n_train, n_eval = len(fit_idx), len(dev_idx)
            else:
                fit_idx = np.concatenate([index[TRAIN], index[DEV]])
                m = _cross_validate(X, y[target], df, agent, s, fit_idx, name, params, seed, order)
                # model cuối cùng (để export / đo holdout) fit trên toàn bộ dữ liệu không thuộc holdout
                model = model_zoo.create(name, seed, params).fit(X.iloc[fit_idx], y[target].iloc[fit_idx])
                eval_name = "cv" if s["dev_mode"] == "kfold" else "lopo"
                golden = X.iloc[fit_idx].head(N_GOLDEN)
                n_train, n_eval = len(fit_idx), len(fit_idx)

            all_metrics[f"{eval_name}/{key}"] = m
            plot_confusion_matrix(m, f"{eval_name}: {key}", out_dir / f"cm_{eval_name}_{key}.png")
            rows.append((eval_name, target, name, n_train, n_eval, m))
            _print_result(eval_name, key, m)

            if holdout is not None:
                X_h, y_h = holdout
                mh = compute_metrics(y_h[target], model.predict(X_h), order)
                all_metrics[f"holdout/{key}"] = mh
                plot_confusion_matrix(mh, f"holdout: {key}", out_dir / f"cm_holdout_{key}.png")
                rows.append(("holdout", target, name, n_train, len(X_h), mh))
                _print_result("holdout", key, mh)

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

    with open(out_dir / "split.json", "w", encoding="utf-8") as f:
        manifest = split_manifest(groups, index if final else visible)
        json.dump({"groups": manifest, "report": report, "warnings": warnings}, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    _append_experiment_log(out_dir.name, version, agent, rows)
    print(f"Đã lưu vào {out_dir}")
    return all_metrics


def _cross_validate(X, y_target, df, agent, s, pool_idx, name, params, seed, order) -> dict:
    """CV trên pool_idx. Trả về chỉ số tính trên dự đoán out-of-fold, kèm trung bình ± std từng fold."""
    by = "player" if s["dev_mode"] == "lopo" else s["group_by"]
    cv_groups = group_keys(df, by, agent).iloc[pool_idx].reset_index(drop=True)
    X_pool, y_pool = X.iloc[pool_idx].reset_index(drop=True), y_target.iloc[pool_idx].reset_index(drop=True)

    # mỗi dòng của pool được validate đúng một lần -> mọi giá trị đều bị ghi đè bằng dự đoán out-of-fold
    oof = y_pool.copy()
    fold_metrics = []
    for tr, va in cv_folds(cv_groups, s["dev_mode"], s["n_folds"]):
        model = model_zoo.create(name, seed, params).fit(X_pool.iloc[tr], y_pool.iloc[tr])
        pred = model.predict(X_pool.iloc[va])
        oof.iloc[va] = pred
        fold_metrics.append(compute_metrics(y_pool.iloc[va], pred, order))

    m = compute_metrics(y_pool, oof, order)  # confusion matrix gộp của mọi fold
    summary = summarize_folds(fold_metrics)
    m.update({k: summary[k] for k in ("accuracy", "accuracy_std", "macro_f1", "macro_f1_std", "n_folds")})
    m["folds"] = [{"accuracy": f["accuracy"], "macro_f1": f["macro_f1"]} for f in fold_metrics]
    return m


def _load_holdout(cfg, s, df, X, y, index, version, agent):
    if cfg.get("holdout_dirs"):
        df_h = load_data(cfg["holdout_dirs"], version, cfg.get("game_versions"))
        assert_no_overlap(df["match_id"], df_h["match_id"])
        return build(df_h, version, agent)
    if s["holdout_ratio"] > 0:
        idx = index[HOLDOUT]
        return X.iloc[idx], y.iloc[idx]
    raise ValueError("Config này không có holdout (holdout_ratio = 0 và không có holdout_dirs) -> bỏ --final")


def _print_split(version, agent, s, report, warnings, holdout, has_holdout: bool):
    names = {TRAIN: "train" if s["dev_mode"] == "single" else "dữ liệu cross-validation"}
    parts = [f"{names.get(n, n)} {r['groups']} nhóm, {r['frames']} frame" for n, r in report.items()]
    print(f"Schema {version} | agent {agent} | group_by {s['group_by']} | dev_mode {s['dev_mode']}")
    print("  " + " / ".join(parts))
    if holdout is not None:
        print(f"  holdout: {len(holdout[0])} frame (đang đo vì --final)")
    elif has_holdout:
        print("  holdout được giữ kín - chỉ đo khi chạy với --final")
    else:
        print("  config này không có holdout")
    for w in warnings:
        print(f"  ! {w}")


def _print_result(eval_name, key, m):
    std = f" ± {m['macro_f1_std']:.3f}" if "macro_f1_std" in m else ""
    print(f"  [{eval_name:<7}] {key:<32} acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}{std}")
    if m["rare_labels"]:
        print(f"            ! lớp ít mẫu (< 10), F1 rất nhiễu: {m['rare_labels']}")


def _fmt(m: dict, key: str) -> str:
    std = m.get(f"{key}_std")
    return f"{m[key]:.3f} ± {std:.3f}" if std is not None else f"{m[key]:.3f}"


def _append_experiment_log(run_id, version, agent, rows) -> None:
    log = REPORTS_DIR / "experiments.md"
    today = datetime.now().strftime("%Y-%m-%d")
    with open(log, "a", encoding="utf-8") as f:
        for eval_name, target, name, n_train, n_eval, m in rows:
            f.write(
                f"| {today} | {run_id} | {version} | {agent} | {target} | {name} "
                f"| {eval_name} | {n_train} | {n_eval} | {_fmt(m, 'accuracy')} | {_fmt(m, 'macro_f1')} | |\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="đường dẫn file YAML trong configs/")
    parser.add_argument("--final", action="store_true", help="đo thêm trên holdout (chỉ khi đã chốt model)")
    args = parser.parse_args()
    run(args.config, final=args.final)


if __name__ == "__main__":
    main()
