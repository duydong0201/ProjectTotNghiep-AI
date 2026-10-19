"""DAgger co giup bot san phai khong: de rieng tung tran DAgger that, train 2 model tren phan con lai
(khong DAgger / co DAgger trong so x20), cham tren frame nguoi gianh quyen theo cach bot ra quyet
dinh trong game (policy.py). Chay: python scripts/eval_dagger_holdout.py
"""

import sys

sys.path.insert(0, "src")
import numpy as np

from spike_ai.policy import (
    action_distribution,
    add_last_touch_kind,
    event_recall,
    expected_confusion,
    policy_params,
    scores_from_confusion,
)
from spike_ai.schema import load_spec
from spike_ai.train import _fit, build_rows, load_config, load_data

cfg = load_config("configs/dagger_v1_w_aligned.yaml")
classes = [str(c) for c in load_spec("v1")["labels"]["action"]["values"]]
df = add_last_touch_kind(load_data(cfg["data_dirs"], "v1", None))
real = [m for m in df.match_id.unique() if m.startswith("2026-10-15_22-")]
print("tran DAgger that:", real)
fitA, XA, yA = build_rows(df, "v1", "player", cfg)  # khong DAgger
fitB, XB, yB = build_rows(df, "v1", ["player", "opponent"], cfg)  # co DAgger, trong so x20
W = fitB["_weight"].to_numpy()
ev, Xe, ye = build_rows(df, "v1", "opponent", {**cfg, "frames_around_action": None})
params = policy_params(cfg)
rf = cfg["models"]["random_forest"]
agg = {"khong DAgger": [], "co DAgger x20": []}
for m in real:
    va = (ev.match_id == m).to_numpy()
    for name, rows, X, y, w in (("khong DAgger", fitA, XA, yA, None), ("co DAgger x20", fitB, XB, yB, W)):
        keep = (rows.match_id != m).to_numpy()
        model = _fit("random_forest", 42, rf, X[keep], y["action"][keep], None if w is None else w[keep])
        p = np.zeros((va.sum(), len(classes)))
        for j, c in enumerate(model.classes_):
            p[:, classes.index(str(c))] = model.predict_proba(Xe[va])[:, j]
        d = action_distribution(ev[va], p, classes, "opponent", params)
        agg[name].append((ye["action"][va].astype(str).to_numpy(), d, ev.match_id[va].to_numpy()))
    print("  xong", m, int(va.sum()), "frame nguoi gianh quyen")
for name, parts in agg.items():
    y = np.concatenate([a for a, _, _ in parts])
    d = np.vstack([b for _, b, _ in parts])
    ids = np.concatenate([c for _, _, c in parts])
    s = scores_from_confusion(
        expected_confusion(y, d, classes), classes, [c for c in classes if c != "None"] + ["None"]
    )
    er = event_recall(y, d, classes, ids, 3)
    print(f"\n=== {name}: macro-F1 bot trong game {s['macro_f1']:.3f}")
    for c in ("Set", "Bump", "SpikeLight", "SpikeMedium", "SpikeStrong", "Jump"):
        if c in er:
            print(f"   {c:<12} bat dung +/-3 frame {er[c]['recall'] * 100:5.1f}%  (n={er[c]['support']})")
