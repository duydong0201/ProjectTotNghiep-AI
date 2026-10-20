"""Đo lại các model trên một BỘ TRẬN ĐO MỚI (seed chưa từng dùng để chọn model).

Vì sao: vòng tự học chọn model cao nhất trên cùng một bộ 50 trận đo (seed 100000), nên con số của model
được chọn bị thổi phồng (vừa tốt vừa gặp may với đúng bộ đó). Đo trên seed khác cho con số thật.

Cách dùng (sửa danh sách MODELS cho đúng đường dẫn):
    python scripts/eval_fresh_seeds.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from spike_ai import selfplay  # noqa: E402
from spike_ai import train as train_module  # noqa: E402

SCRATCH = Path("scratch/fresh_eval")  # trong .gitignore
SCRATCH.mkdir(parents=True, exist_ok=True)
GAME = Path("F:/DoAnTotNghiep/ProjectTotNghiep")
START_COMMIT = "6e318b8"  # commit game chua model dagger_v1_w_aligned (truoc khi tu hoc)
start_header = SCRATCH / "model_start.h"
if not start_header.exists():
    blob = subprocess.run(
        ["git", "-C", str(GAME), "show", f"{START_COMMIT}:Source/AI/Generated/spike_ai_action_random_forest.h"],
        capture_output=True,
        check=True,
    ).stdout
    start_header.write_bytes(blob)

MODELS = {
    "xuat phat (dagger_v1_w_aligned)": start_header,
    "tot nhat v1 (dang o game chinh)": Path(
        "models/selfplay_expert_v1_20261020-123720/best/spike_ai_action_random_forest.h"
    ),
    "v2 vong 3 (69,2% tren seed cu)": Path(
        "models/selfplay_expert_v2_20261020-142907/iter03/spike_ai_action_random_forest.h"
    ),
}
JUMP = {"nhay 31,4%": [], "luon nhay": ["--jump-after-set", "1.0"]}
SEED, MATCHES = 200000, 50

sp = train_module.load_config("configs/selfplay_expert_v1.yaml")["selfplay"]
sp = {**sp, "eval_seed": SEED, "eval_matches": MATCHES}
keep = selfplay.game_paths(sp)["auto_header"]
backup = SCRATCH / "auto_header_backup.h"
shutil.copyfile(keep, backup)
rows = []
try:
    for name, header in MODELS.items():
        selfplay.install_header(header, sp)
        selfplay.build_runner(sp)
        for jname, args in JUMP.items():
            r = selfplay.score_ml({**sp, "runner_args": args}, SCRATCH / "eval")
            rows.append((name, jname, r))
            print(
                f"{name:<34} {jname:<11} {r['pct_diem_thang']:5.1f}%  ({r['diem_thang']}-{r['diem_mat']})", flush=True
            )
finally:
    selfplay.install_header(backup, sp)
    selfplay.build_runner(sp)
    print("da tra GeneratedAuto ve nhu cu")
