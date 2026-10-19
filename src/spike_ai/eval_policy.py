"""Đo offline theo ĐÚNG cách bot ra quyết định trong game, trên MỌI frame (xem policy.py).

Cách dùng:
    python -m spike_ai.eval_policy --config configs/played_v1_aligned.yaml
    python -m spike_ai.eval_policy --config ... --model random_forest --tolerance 3

Cross-validation theo trận như `train.py` (cùng config, cùng holdout bị loại ra), nhưng:
    - model fit trên đúng các dòng train.py dùng (lọc người điều khiển, label_shift, cửa sổ);
    - chấm điểm trên MỌI frame của trận để riêng, vì game hỏi model ở mọi frame, không chỉ
      quanh lúc chạm bóng.

In ra ba cách đo cạnh nhau, trên cùng các fold:
    argmax / cửa sổ    cách train.py đang báo cáo
    argmax / mọi frame cùng model, nhưng đo ở mọi frame
    bot trong game     ngưỡng + rút thăm + luật chặn + luật nhảy, ở mọi frame
"""

import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd

from .paths import MODELS_DIR, REPORTS_DIR
from .policy import (
    action_distribution,
    add_last_touch_kind,
    event_recall,
    expected_confusion,
    policy_params,
    scores_from_confusion,
)
from .schema import load_spec
from .split import HOLDOUT, cv_folds, group_keys, split_by_hash
from .train import (
    _fit,
    action_window_mask,
    as_agent_list,
    build_rows,
    check_label_shift,
    load_config,
    load_data,
    validate_split_config,
)

TARGET = "action"
METHODS = ("argmax / cửa sổ", "argmax / mọi frame", "bot trong game")


def _onehot(pred, classes) -> np.ndarray:
    out = np.zeros((len(pred), len(classes)))
    index = {c: k for k, c in enumerate(classes)}
    out[np.arange(len(pred)), [index[str(p)] for p in pred]] = 1.0
    return out


def _align(proba, model_classes, classes) -> np.ndarray:
    """Cột xác suất theo thứ tự `classes` (fold có thể thiếu lớp hiếm)."""
    out = np.zeros((len(proba), len(classes)))
    for j, c in enumerate(model_classes):
        out[:, classes.index(str(c))] = proba[:, j]
    return out


def run(config_path: str, model_name: str | None = None, tolerance: int = 3, save: bool = True) -> dict:
    cfg = load_config(config_path)
    version, seed = cfg["schema"], cfg.get("seed", 42)
    agents = as_agent_list(cfg["agent"])
    if len(agents) != 1:
        raise ValueError("eval_policy đo cho MỘT agent (bot điều khiển một nhân vật)")
    agent = agents[0]
    if model_name is None:
        model_name = "random_forest" if "random_forest" in cfg["models"] else next(iter(cfg["models"]))
    params_model = cfg["models"][model_name] or {}
    params = policy_params(cfg)
    s = validate_split_config(cfg)
    check_label_shift(cfg, version)
    classes = [str(c) for c in load_spec(version)["labels"][TARGET]["values"]]

    df = load_data(cfg["data_dirs"], version, cfg.get("game_versions"))
    excluded = set(cfg.get("exclude_matches") or [])
    if excluded:
        df = df[~df["match_id"].isin(excluded)].reset_index(drop=True)
    df = add_last_touch_kind(df)

    print("Dòng để fit (như train.py):")
    fit_rows, X_fit, y_fit = build_rows(df, version, agent, cfg)
    print("Dòng để chấm (mọi frame):")
    eval_rows, X_eval, y_eval = build_rows(df, version, agent, {**cfg, "frames_around_action": None})
    W = fit_rows["_weight"].to_numpy() if cfg.get("agent_weight") else None

    # Loại holdout y như train.py: không bao giờ đụng tới.
    groups = group_keys(eval_rows, s["group_by"], agent)
    holdout = split_by_hash(groups, s["salt"], s["dev_ratio"], s["holdout_ratio"])[HOLDOUT]
    pool = np.setdiff1d(np.arange(len(eval_rows)), holdout)
    holdout_matches = set(eval_rows["match_id"].iloc[holdout])

    window = cfg.get("frames_around_action")
    in_window = action_window_mask(eval_rows, y_eval, int(window)) if window else np.ones(len(eval_rows), bool)

    y = y_eval[TARGET].astype(str).to_numpy()
    dist = {m: np.zeros((len(eval_rows), len(classes))) for m in METHODS}
    by = "player" if s["dev_mode"] == "lopo" else s["group_by"]
    mode = "lopo" if s["dev_mode"] == "lopo" else "kfold"
    cv_groups = group_keys(eval_rows, by, agent).iloc[pool].reset_index(drop=True)

    folds = cv_folds(cv_groups, mode, s["n_folds"])
    for n, (_, va) in enumerate(folds, 1):
        va = pool[va]
        val_matches = set(eval_rows["match_id"].iloc[va])
        fit = (~fit_rows["match_id"].isin(val_matches | holdout_matches)).to_numpy()
        model = _fit(model_name, seed, params_model, X_fit[fit], y_fit[TARGET][fit], None if W is None else W[fit])
        proba = _align(model.predict_proba(X_eval.iloc[va]), model.classes_, classes)
        top = _onehot(np.asarray(classes)[proba.argmax(axis=1)], classes)
        dist["argmax / mọi frame"][va] = top
        dist["argmax / cửa sổ"][va] = top
        dist["bot trong game"][va] = action_distribution(eval_rows.iloc[va], proba, classes, agent, params)
        print(f"  fold {n}/{len(folds)}: fit {int(fit.sum()):,} dòng, chấm {len(va):,} frame")

    scored = {
        "argmax / cửa sổ": np.intersect1d(pool, np.flatnonzero(in_window)),
        "argmax / mọi frame": pool,
        "bot trong game": pool,
    }
    order = [c for c in classes if c != "None"] + ["None"]
    report = {"config": config_path, "model": model_name, "policy": params, "tolerance": tolerance, "methods": {}}
    for method, rows in scored.items():
        cm = expected_confusion(y[rows], dist[method][rows], classes)
        result = scores_from_confusion(cm, classes, order)
        result["frames"] = int(len(rows))
        result["event_recall"] = event_recall(
            y[rows], dist[method][rows], classes, eval_rows["match_id"].to_numpy()[rows], tolerance
        )
        report["methods"][method] = result

    n_matches = eval_rows["match_id"].iloc[pool].nunique()
    report["matches"] = int(n_matches)
    report["actions_per_match"] = {
        c: {
            "nguoi_choi": float((y[pool] == c).sum() / n_matches),
            "bot_ky_vong": float(dist["bot trong game"][pool, classes.index(c)].sum() / n_matches),
        }
        for c in classes
        if c != "None"
    }
    _print_report(report, classes)

    if save:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = MODELS_DIR / f"{cfg['run_name']}_policy_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "policy_metrics.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        _append_experiment_log(out_dir.name, version, agent, model_name, report, len(X_fit))
        print(f"Đã lưu vào {out_dir}")
    return report


def _print_report(report: dict, classes) -> None:
    methods = report["methods"]
    print(f"\n{report['matches']} trận, model {report['model']}, policy {report['policy']}")
    print("\nF1 từng lớp (chấm theo đúng frame):")
    print(f"{'lớp':<13}" + "".join(f"{m:>22}" for m in METHODS))
    for c in [c for c in classes if c != "None"] + ["None"]:
        cells = []
        for m in METHODS:
            pc = methods[m]["per_class"].get(c)
            cells.append(f"{pc['f1']:.3f}" if pc and pc["support"] > 0 else "-")
        print(f"{c:<13}" + "".join(f"{v:>22}" for v in cells))
    print(f"{'macro-F1':<13}" + "".join(f"{methods[m]['macro_f1']:>22.3f}" for m in METHODS))
    print(f"{'số frame':<13}" + "".join(f"{methods[m]['frames']:>22,}" for m in METHODS))

    tol = report["tolerance"]
    print(f"\nTỉ lệ cú của người chơi mà bot cũng ra đúng cú đó trong +/-{tol} frame:")
    print(f"{'lớp':<13}{'argmax / mọi frame':>22}{'bot trong game':>22}")
    for c in classes:
        if c == "None":
            continue
        a = methods["argmax / mọi frame"]["event_recall"].get(c)
        b = methods["bot trong game"]["event_recall"].get(c)
        if a or b:
            print(f"{c:<13}{(a or {}).get('recall', 0):>22.3f}{(b or {}).get('recall', 0):>22.3f}")

    print("\nSố lần ra tay mỗi trận (bot: kỳ vọng theo frame, chưa tính cooldown nên cao hơn trong game):")
    print(f"{'lớp':<13}{'người chơi':>14}{'bot':>14}")
    for c, v in report["actions_per_match"].items():
        print(f"{c:<13}{v['nguoi_choi']:>14.1f}{v['bot_ky_vong']:>14.1f}")


def _append_experiment_log(run_id, version, agent, model_name, report, n_fit) -> None:
    m = report["methods"]
    p = report["policy"]
    today = datetime.now().strftime("%Y-%m-%d")
    note = (
        f"eval_policy (do theo cach bot ra quyet dinh trong game, nguong {p['act_threshold']}, T={p['temperature']}). "
        f"macro-F1: argmax/cua so {m['argmax / cửa sổ']['macro_f1']:.3f}, "
        f"argmax/moi frame {m['argmax / mọi frame']['macro_f1']:.3f}, "
        f"bot trong game {m['bot trong game']['macro_f1']:.3f}"
    )
    with open(REPORTS_DIR / "experiments.md", "a", encoding="utf-8") as f:
        f.write(
            f"| {today} | {run_id} | {version} | {agent} | {TARGET} | {model_name} | policy-cv "
            f"| {n_fit} | {m['bot trong game']['frames']} | - | {m['bot trong game']['macro_f1']:.3f} | {note} |\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="file YAML trong configs/")
    parser.add_argument("--model", default=None, help="tên model trong config (mặc định random_forest)")
    parser.add_argument("--tolerance", type=int, default=3, help="số frame lệch vẫn tính là đúng lúc")
    args = parser.parse_args()
    run(args.config, args.model, args.tolerance)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
