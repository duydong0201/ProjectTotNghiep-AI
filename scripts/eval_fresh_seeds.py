"""Đo lại các model trên một BỘ TRẬN ĐO MỚI (seed chưa từng dùng để chọn model).

Vì sao: vòng tự học chọn model cao nhất trên một bộ trận đo, nên con số của model được chọn bị thổi
phồng (vừa tốt vừa gặp may với đúng bộ đó). Đo trên seed khác cho con số thật (ADR-015).

Cách dùng:
    python scripts/eval_fresh_seeds.py --seed 400000 \\
        --model "xuat phat=game:6e318b8" \\
        --model "v3=models/selfplay_expert_v3_.../best/spike_ai_action_random_forest.h"

`--model ten=duong_dan` (file .h) hoặc `ten=game:<commit>` (lấy file model ở commit đó của repo game).
Mỗi model đo với hai luật nhảy: 31,4% (mặc định của game) và luôn nhảy sau Set.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")
from spike_ai import selfplay  # noqa: E402
from spike_ai import train as train_module  # noqa: E402

SCRATCH = Path("scratch/fresh_eval")  # trong .gitignore
GAME = Path("F:/DoAnTotNghiep/ProjectTotNghiep")
GAME_MODEL = "Source/AI/Generated/spike_ai_action_random_forest.h"
JUMP = {"nhay 31,4%": [], "luon nhay": ["--jump-after-set", "1.0"]}


def resolve_model(spec: str, name: str) -> Path:
    if not spec.startswith("game:"):
        return Path(spec)
    out = SCRATCH / f"model_{name}.h"
    blob = subprocess.run(
        ["git", "-C", str(GAME), "show", f"{spec[5:]}:{GAME_MODEL}"], capture_output=True, check=True
    ).stdout
    out.write_bytes(blob)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--matches", type=int, default=50)
    parser.add_argument("--model", action="append", required=True, help="ten=duong_dan hoac ten=game:<commit>")
    args = parser.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    models = {}
    for item in args.model:
        name, spec = item.split("=", 1)
        models[name] = resolve_model(spec, name.replace(" ", "_"))

    sp = train_module.load_config("configs/selfplay_expert_v1.yaml")["selfplay"]
    sp = {**sp, "eval_seed": args.seed, "eval_matches": args.matches}
    keep = selfplay.game_paths(sp)["auto_header"]
    backup = SCRATCH / "auto_header_backup.h"
    shutil.copyfile(keep, backup)
    try:
        for name, header in models.items():
            selfplay.install_header(header, sp)
            selfplay.build_runner(sp)
            for jname, extra in JUMP.items():
                r = selfplay.score_ml({**sp, "runner_args": extra}, SCRATCH / "eval")
                pct, won, lost = r["pct_diem_thang"], r["diem_thang"], r["diem_mat"]
                print(f"{name:<24} {jname:<11} {pct:5.1f}%  ({won}-{lost})", flush=True)
    finally:
        selfplay.install_header(backup, sp)
        selfplay.build_runner(sp)
        print("da tra GeneratedAuto ve nhu cu")


if __name__ == "__main__":
    main()
