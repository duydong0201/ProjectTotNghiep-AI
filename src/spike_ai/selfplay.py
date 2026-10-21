"""Vòng TỰ HỌC tự động: bot ML tự chơi, tự thu dữ liệu, tự train, chỉ giữ model nào tốt hơn.

Cách dùng:
    python -m spike_ai.selfplay --config configs/selfplay_v1.yaml
    python -m spike_ai.selfplay --config configs/selfplay_v1.yaml --iterations 1   # chạy thử 1 vòng

Mỗi vòng (không cần người chơi):
    1. MatchRunner (repo game, không cửa sổ) cho bot luật (trái) đấu bot ML (phải) N trận, ghi log
       vào Logs/auto/iter_XX/ - KHÔNG trộn với Logs/v1 (dữ liệu người chơi).
    2. Train model mới trên Logs/v1 + toàn bộ Logs/auto. Phía bot ML học từ chính nó (self-imitation,
       ADR-014): chỉ những pha bóng nó thắng điểm (train.build_rows, `self_imitation`).
    3. Export header, chép vào Source/AI/GeneratedAuto/ của game, build lại MatchRunner.
    4. Đo: bot ML mới đấu bot luật trên CÙNG bộ seed đo như model đang giữ, tính % điểm thắng.
    5. Tốt hơn model đang giữ thì giữ; kém hơn thì trả lại model cũ.

Game chính (MyGame.exe, Source/AI/Generated/) KHÔNG bị đụng tới. Model tốt nhất nằm ở
models/<run>/best/; đưa vào game chính là việc làm tay, sau khi đã xem kết quả.
"""

import argparse
import gzip
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from . import eval_ingame, export_cpp
from . import train as train_module
from .paths import MODELS_DIR, REPORTS_DIR, resolve

HEADER_NAME = "spike_ai_action_random_forest.h"
TRAIN_KEYS_DROP = ("selfplay",)


# ------------------------------------------------------------------ các bước chạm tới repo game
def game_paths(sp: dict) -> dict:
    game = Path(sp["game_dir"])
    return {
        "runner": game / sp.get("runner", "build/RelWithDebInfo/MatchRunner.exe"),
        "build_dir": game / sp.get("build_dir", "build"),
        "auto_header": game / "Source/AI/GeneratedAuto" / HEADER_NAME,
        "game_header": game / "Source/AI/Generated" / HEADER_NAME,
    }


def run_matches(sp: dict, out_dir: Path, matches: int, seed: int, tag: str, expert_prob: float = 0.0) -> str:
    """Gọi MatchRunner, trả về dòng TONG của nó. `expert_prob` > 0: chuyên gia điều khiển một phần lượt (ADR-015)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(game_paths(sp)["runner"]),
        "--out", str(out_dir),
        "--matches", str(matches),
        "--max-frames", str(sp.get("max_frames", 54000)),
        "--seed", str(seed),
        "--tag", tag,
        "--expert-prob", str(expert_prob),
    ]  # fmt: skip
    # Tham số thêm cho MỌI lần gọi (tự chơi lẫn đo), vd luật nhảy: ["--jump-after-set", "1.0"].
    cmd += [str(a) for a in sp.get("runner_args", [])]
    done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    totals = [line for line in done.stdout.splitlines() if line.startswith("TONG")]
    return totals[-1] if totals else done.stdout.strip().splitlines()[-1]


def compress_logs(folder: Path) -> None:
    """Nén mọi .csv trong thư mục thành .csv.gz (nhỏ hơn ~40 lần, pandas đọc thẳng được) rồi xoá bản gốc."""
    for csv in sorted(folder.rglob("*.csv")):
        with open(csv, "rb") as src, gzip.open(csv.with_name(csv.name + ".gz"), "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        csv.unlink()


def build_runner(sp: dict) -> None:
    """Build lại MatchRunner (chỉ target này, không đụng MyGame)."""
    paths = game_paths(sp)
    # Qua file .bat: truyền chuỗi có dấu nháy thẳng cho `cmd /c` từ Python bị escape sai.
    bat = paths["build_dir"] / "_selfplay_build.bat"
    bat.write_text(
        "@echo off\r\n"
        f'call "{Path(sp["vcvars"])}" >nul\r\n'
        f'cmake --build "{paths["build_dir"]}" --config RelWithDebInfo --target MatchRunner -- /v:m\r\n',
        encoding="utf-8",
    )
    done = subprocess.run([str(bat)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        tail = "\n".join((done.stdout + done.stderr).splitlines()[-20:])
        raise RuntimeError(f"Build MatchRunner thất bại:\n{tail}")


def install_header(header: Path, sp: dict) -> None:
    target = game_paths(sp)["auto_header"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(header, target)


# ------------------------------------------------------------------ train và đo
def recent_selfplay_dirs(auto_logs: Path, keep: int) -> list[Path]:
    """`keep` thư mục vòng tự chơi gần nhất (tên có thời gian chạy + số vòng nên sắp theo tên là đúng thứ tự)."""
    # CHỈ thư mục vòng tự chơi (iter_*). Thư mục khác trong Logs/auto, vd xem_truc_tiep (run_botvsbot.bat),
    # không phải dữ liệu tự học - và file của trận đang xem có thể đang ghi dở.
    folders = sorted(p for p in auto_logs.glob("iter_*") if p.is_dir()) if auto_logs.exists() else []
    return folders[-keep:] if keep > 0 else folders


def train_config(cfg: dict, run_dir: Path, iteration: int) -> Path:
    """Config train của vòng này = config gốc bỏ mục `selfplay`, đặt run_name theo vòng.

    `selfplay.keep_last_iterations` = K > 0: chỉ dùng log tự chơi của K vòng gần nhất thay cho cả
    thư mục Logs/auto, để dữ liệu tự học không lấn át dữ liệu người chơi khi chạy nhiều vòng.
    """
    train_cfg = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS_DROP}
    train_cfg["run_name"] = f"{cfg['run_name']}_iter{iteration:02d}"
    sp = cfg.get("selfplay", {})
    keep = int(sp.get("keep_last_iterations", 0))
    if keep and sp.get("auto_logs"):
        auto = Path(sp["auto_logs"]).resolve()
        others = [d for d in train_cfg["data_dirs"] if Path(d).resolve() != auto]
        train_cfg["data_dirs"] = others + [str(p) for p in recent_selfplay_dirs(auto, keep)]
    path = run_dir / f"train_iter{iteration:02d}.yaml"
    path.write_text(yaml.safe_dump(train_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def train_candidate(config_path: Path, run_name: str) -> Path:
    """Train, trả về file .joblib của model action vừa train."""
    before = set(MODELS_DIR.glob(f"{run_name}_*"))
    train_module.run(str(config_path))
    new = sorted(set(MODELS_DIR.glob(f"{run_name}_*")) - before)
    if not new:
        raise RuntimeError(f"Không thấy thư mục model mới cho {run_name}")
    model = new[-1] / "action__random_forest.joblib"
    if not model.exists():
        raise RuntimeError(f"{model} không tồn tại - config phải train target action bằng random_forest")
    return model


def score_ml(sp: dict, eval_dir: Path) -> dict:
    """Bot ML (sân phải) đấu bot luật trên bộ seed đo cố định. Trả về % điểm thắng và vài chỉ số."""
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    run_matches(sp, eval_dir, sp.get("eval_matches", 20), sp.get("eval_seed", 100000), "eval")
    right = eval_ingame.evaluate([eval_dir])["sides"]["phải"]
    # Log trận đo chỉ dùng để tính điểm một lần (không dùng train) và rất nặng (~8 MB/trận): xoá.
    shutil.rmtree(eval_dir, ignore_errors=True)
    keys = ("pct_diem_thang", "diem_thang", "diem_mat", "cu_dap_trung_moi_tran", "set_moi_tran", "jump_moi_tran")
    return {k: right[k] for k in keys} | {"chuoi_set_jump_spike": right["chuoi_set_jump_spike"]}


def decide(best_pct: float, candidate_pct: float, min_gain: float) -> bool:
    """Giữ model mới khi % điểm thắng cao hơn model đang giữ ít nhất `min_gain` điểm phần trăm."""
    return candidate_pct > best_pct + min_gain


# ------------------------------------------------------------------ vòng lặp
def run(
    config_path: str,
    iterations: int | None = None,
    max_minutes: float | None = None,
    stop_after: int | None = None,
) -> dict:
    cfg = train_module.load_config(config_path)
    sp = cfg["selfplay"]
    n_iter = iterations if iterations is not None else int(sp.get("iterations", 3))
    started = time.monotonic()
    auto_logs = Path(sp["auto_logs"])
    auto_logs.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = MODELS_DIR / f"{cfg['run_name']}_{stamp}"
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    paths = game_paths(sp)

    # Model xuất phát: model đang có ở GeneratedAuto, chưa có thì lấy model của game chính.
    if not paths["auto_header"].exists():
        install_header(paths["game_header"], sp)
    shutil.copyfile(paths["auto_header"], best_dir / HEADER_NAME)
    print("== Đo model xuất phát")
    build_runner(sp)
    best = score_ml(sp, run_dir / "eval" / "start")
    print(f"   bot ML thắng {best['pct_diem_thang']:.1f}% điểm")
    # Bộ trận XÁC NHẬN (confirm_seed): chọn model trên bộ đo chính (eval_seed) rồi kiểm lại trên bộ này,
    # vì chọn cái cao nhất trên CÙNG một bộ đo thì con số được chọn bị thổi phồng (ADR-015).
    confirm_sp = {**sp, "eval_seed": sp["confirm_seed"]} if sp.get("confirm_seed") is not None else None
    best_confirm = score_ml(confirm_sp, run_dir / "eval" / "start_confirm") if confirm_sp else None
    if best_confirm:
        print(f"   bộ xác nhận: {best_confirm['pct_diem_thang']:.1f}% điểm")
    history = [{"iteration": 0, "model": "xuat phat", "eval": best, "confirm": best_confirm, "kept": True}]

    # Seed tự chơi khác nhau giữa CÁC LẦN chạy (cùng model + cùng seed = đúng các trận cũ).
    # Seed đo (eval_seed) thì giữ cố định để mọi model đấu cùng một bộ trận.
    seed_base = int(datetime.now().strftime("%y%m%d%H%M")) % 100_000_000 * 100
    stop_after = int(sp.get("stop_after_rejections", 0)) if stop_after is None else stop_after
    rejected_in_a_row = 0
    for it in range(1, n_iter + 1):
        # Hết giờ (max_minutes) thì không mở vòng mới; vòng đang chạy luôn được làm trọn.
        if max_minutes is not None and (time.monotonic() - started) / 60.0 >= max_minutes:
            print(f"\n== Hết {max_minutes:.0f} phút -> dừng trước vòng {it}")
            break
        print(f"\n== Vòng {it}/{n_iter}")
        iter_logs = auto_logs / f"iter_{stamp}_{it:02d}"
        total = run_matches(
            sp, iter_logs, sp["matches_per_iteration"], seed_base + it, f"it{it:02d}", sp.get("expert_prob", 0.0)
        )
        compress_logs(iter_logs)
        print(f"   tự chơi: {total}")
        cfg_path = train_config(cfg, run_dir, it)
        model = train_candidate(cfg_path, f"{cfg['run_name']}_iter{it:02d}")
        header = export_cpp.export(model, run_dir / f"iter{it:02d}")
        install_header(header, sp)
        build_runner(sp)
        cand = score_ml(sp, run_dir / "eval" / f"iter{it:02d}")
        kept = decide(best["pct_diem_thang"], cand["pct_diem_thang"], float(sp.get("min_gain", 0.0)))
        cand_confirm = None
        if kept and confirm_sp:
            cand_confirm = score_ml(confirm_sp, run_dir / "eval" / f"iter{it:02d}_confirm")
            kept = cand_confirm["pct_diem_thang"] >= best_confirm["pct_diem_thang"]
            new_pct, old_pct = cand_confirm["pct_diem_thang"], best_confirm["pct_diem_thang"]
            print(f"   bộ xác nhận: {new_pct:.1f}% (đang giữ {old_pct:.1f}%)")
        print(f"   model mới thắng {cand['pct_diem_thang']:.1f}% điểm (đang giữ {best['pct_diem_thang']:.1f}%) -> "
              f"{'GIỮ' if kept else 'BỎ'}")  # fmt: skip
        if kept:
            best = cand
            best_confirm = cand_confirm
            shutil.copyfile(header, best_dir / HEADER_NAME)
            (best_dir / "model.txt").write_text(str(model), encoding="utf-8")
        else:
            install_header(best_dir / HEADER_NAME, sp)
            build_runner(sp)
        history.append(
            {
                "iteration": it,
                "model": str(model),
                "self_play": total,
                "eval": cand,
                "confirm": cand_confirm,
                "kept": kept,
            }
        )
        (run_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        rejected_in_a_row = 0 if kept else rejected_in_a_row + 1
        if stop_after and rejected_in_a_row >= stop_after:
            print(f"   {rejected_in_a_row} vòng liền không giữ được model nào -> dừng")
            break

    _log(cfg["run_name"], run_dir.name, history)
    print(f"\nXong. Model tốt nhất: {best_dir}. Lịch sử: {run_dir / 'history.json'}")
    return {"run_dir": str(run_dir), "history": history}


def _log(run_name: str, run_id: str, history: list) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    steps = "; ".join(
        f"vong {h['iteration']}: {h['eval']['pct_diem_thang']:.1f}% {'giu' if h['kept'] else 'bo'}" for h in history
    )
    with open(REPORTS_DIR / "experiments.md", "a", encoding="utf-8") as f:
        f.write(
            f"| {today} | {run_id} | v1 | opponent | action | random_forest | selfplay (bot ML vs bot luat) "
            f"| - | - | - | - | tu hoc tu dong (ADR-014), % diem bot ML thang tren bo seed do co dinh: {steps} |\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=None, help="ghi đè selfplay.iterations")
    parser.add_argument("--max-minutes", type=float, default=None, help="không mở vòng mới sau ngần này phút")
    parser.add_argument(
        "--stop-after", type=int, default=None, help="ghi đè selfplay.stop_after_rejections (0 = không dừng)"
    )
    args = parser.parse_args()
    run(str(resolve(args.config)), args.iterations, args.max_minutes, args.stop_after)


if __name__ == "__main__":
    main()
