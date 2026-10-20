"""Doi chung cho vong tu hoc (chay: python scripts/selfplay_control.py): train y het selfplay_v1 nhung KHONG co du lieu tu hoc, do tren cung bo seed."""

import filecmp
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
from spike_ai import export_cpp, selfplay  # noqa: E402
from spike_ai import train as train_module  # noqa: E402

cfg = train_module.load_config("configs/selfplay_v1.yaml")
sp = cfg["selfplay"]
cfg = {k: v for k, v in cfg.items() if k not in ("self_imitation",)}
cfg["data_dirs"] = [d for d in cfg["data_dirs"] if not d.rstrip("/").endswith("Logs/auto")]
cfg["selfplay"] = {**sp, "keep_last_iterations": 0}
cfg["run_name"] = "selfplay_control"

run_dir = Path("models/selfplay_control")
run_dir.mkdir(parents=True, exist_ok=True)
paths = selfplay.game_paths(sp)
baseline = run_dir / "baseline.h"
shutil.copyfile(paths["auto_header"], baseline)

cfg_path = selfplay.train_config(cfg, run_dir, 0)
print("data_dirs:", train_module.load_config(cfg_path)["data_dirs"])
model = selfplay.train_candidate(cfg_path, "selfplay_control_iter00")
header = export_cpp.export(model, run_dir / "export")
print("header giong het model xuat phat:", filecmp.cmp(header, baseline, shallow=False))

selfplay.install_header(header, sp)
selfplay.build_runner(sp)
score = selfplay.score_ml(sp, run_dir / "eval")
print(f"DOI CHUNG (khong tu hoc): bot ML thang {score['pct_diem_thang']:.1f}% diem", score)

selfplay.install_header(baseline, sp)
selfplay.build_runner(sp)
print("da tra model xuat phat ve GeneratedAuto")
