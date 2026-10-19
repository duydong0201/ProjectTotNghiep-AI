"""Đo bot TRONG GAME từ log trận (Logs/v1/*.csv): các chỉ số báo cáo cho từng phía sân.

Cách dùng:
    python -m spike_ai.eval_ingame F:/DoAnTotNghiep/ProjectTotNghiep/Logs/v1/2026-10-15_17-19-18_P01.csv
    python -m spike_ai.eval_ingame <thư mục hoặc nhiều file> --json ket_qua.json

Mỗi file là một trận. Phía sân được gọi theo ai điều khiển (cột p_/o_controller): người, bot luật,
bot ML. Dùng được cho người vs bot ML, bot luật vs bot ML (bot-vs-bot) và buổi DAgger.

Chỉ số (cùng tên với BAO-CAO-TIEN-DO-2.md để so được):
    điểm thắng, % điểm thắng, điểm mất mỗi trận
    hành động mỗi trận        số lần BẮT ĐẦU một intent (giữ phím nhiều frame vẫn tính 1)
    cú đập mỗi trận           bấm: số lần ra lệnh đập; trúng: số cú game thật sự chấp nhận
    tỉ trọng tấn công         cú đập trúng / mọi cú chạm trúng
    tỉ lệ đứng yên            frame bóng đang sống mà không di chuyển, không làm gì
    tỉ lệ bỏ lỡ bóng vào tầm  lần bóng lọt vào vùng đập (< 50%) mà không chạm được
    Set / Bump / Jump, chuỗi Set -> Jump -> Spike, Jump bừa (không sau cú Set/Bump của mình,
    không phải lúc giao bóng), phân bố 3 cú đập, cách mất điểm, pha bóng dài nhất

"Trúng" lấy từ RallyState (người chạm cuối / số chạm đổi), không phải từ phím bấm: game bỏ qua
cú đánh khi bóng ngoài vùng detect, nên số lần bấm cao hơn số cú thật.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ENTITY_NONE, is_feature_dump, load_match, log_files, rebuild_kind_v1_to_v2
from .policy import MIN_Y, detect_percent
from .schema import load_spec

SIDES = {  # tên agent trong policy.py, tiền tố cột, entity, đội
    "trái": ("player", "p_", 0, 0),
    "phải": ("opponent", "o_", 3, 1),
}
CONTROLLER_NAMES = {0: "người", 1: "bot luật", 2: "bot ML"}
ALIVE = -1


def _enums():
    spec = load_spec("v1")["enums"]
    intent = {v: k for k, v in spec["FinalIntent"].items()}
    state = {v: k for k, v in spec["ActionState"].items()}
    return intent, state


INTENT_NAME, STATE_NAME = _enums()
SPIKE_INTENTS = {"Spike", "SpikeLight", "SpikeMedium", "SpikeStrong"}
SPIKE_STATES = SPIKE_INTENTS
SPIKE_KINDS = ("SpikeLight", "SpikeMedium", "SpikeStrong")


def controller_of(df: pd.DataFrame, prefix: str) -> str:
    """Ai điều khiển phía sân này. Có nhiều loại (buổi DAgger) thì ghép lại, vd 'bot ML + người'."""
    col = f"{prefix}controller"
    if col not in df:
        return "?"
    counts = df[col].fillna(-1).astype(int).value_counts()
    return " + ".join(CONTROLLER_NAMES.get(int(k), str(k)) for k in counts.index)


def touch_events(df: pd.DataFrame) -> pd.DataFrame:
    """Các cú chạm game đã chấp nhận: (dòng, entity, loại cú)."""
    lt = df["rally_last_touch"].fillna(ENTITY_NONE).astype(int).to_numpy()
    tc = df["rally_touch_count"].fillna(0).astype(int).to_numpy()
    if "rally_last_touch_kind" in df:
        kind = df["rally_last_touch_kind"].fillna(0).astype(int).to_numpy()
    else:
        kind = rebuild_kind_v1_to_v2(df).to_numpy()
    new = np.r_[False, (lt[1:] != lt[:-1]) | (tc[1:] != tc[:-1])] & (lt != ENTITY_NONE)
    idx = np.flatnonzero(new)
    return pd.DataFrame({"row": idx, "entity": lt[idx], "kind": [STATE_NAME.get(int(k), str(k)) for k in kind[idx]]})


def intent_onsets(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Các lần BẮT ĐẦU một intent (khác None và khác frame trước)."""
    it = df[f"{prefix}input_intent"].fillna(0).astype(int).to_numpy()
    start = np.flatnonzero((it != 0) & np.r_[True, it[1:] != it[:-1]])
    return pd.DataFrame({"row": start, "intent": [INTENT_NAME.get(int(v), str(v)) for v in it[start]]})


def rallies(df: pd.DataFrame) -> list[tuple[int, int]]:
    """(dòng đầu, dòng cuối) của từng đoạn bóng đang sống."""
    alive = (df["ball_state_frame"] == ALIVE).to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, alive.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2] - 1, strict=True))


def side_metrics(df: pd.DataFrame, side: str) -> dict:
    """Số đếm thô của một phía sân trong một trận (cộng dồn được qua nhiều trận)."""
    agent, prefix, entity, team = SIDES[side]
    other = SIDES["phải" if side == "trái" else "trái"]
    alive = (df["ball_state_frame"] == ALIVE).to_numpy()
    touches = touch_events(df)
    mine = touches[touches["entity"] == entity]
    onsets = intent_onsets(df, prefix)

    # ---- điểm
    my_score = df["score_left" if team == 0 else "score_right"].to_numpy()
    their_score = df["score_right" if team == 0 else "score_left"].to_numpy()
    won = int(np.clip(np.diff(my_score), 0, None).sum())
    lost_at = np.flatnonzero(np.diff(their_score) > 0) + 1

    # ---- đứng yên
    move = df[f"{prefix}input_move"].fillna(0).to_numpy()
    intent = df[f"{prefix}input_intent"].fillna(0).to_numpy()
    idle = int((alive & (move == 0) & (intent == 0)).sum())

    # ---- bỏ lỡ bóng vào tầm (cú đập < 50%)
    pct = detect_percent(df[f"{prefix}x"], df[f"{prefix}y"], df["ball_x"], df["ball_y"], agent)
    in_zone = alive & (pct < 50.0)
    touch_rows = set(mine["row"].tolist())
    zone_episodes = missed = 0
    edges = np.flatnonzero(np.diff(np.r_[0, in_zone.astype(int), 0]))
    for start, end in zip(edges[::2], edges[1::2], strict=True):
        zone_episodes += 1
        if not any(r in touch_rows for r in range(start, min(end + 2, len(df)))):
            missed += 1

    # ---- Jump: có hợp lệ không, và chuỗi Set -> Jump -> Spike
    y = df[f"{prefix}y"].to_numpy(float)
    rows = onsets["row"].to_numpy()
    on_ground = y[np.maximum(rows - 1, 0)] <= MIN_Y  # chỉ tính lúc đang đứng đất
    jumps = onsets[(onsets["intent"] == "Jump").to_numpy() & on_ground]
    serve_turn = ((df["ball_state_frame"] != ALIVE) & (df["serving_team"] == team)).to_numpy()
    random_jumps = chains = 0
    for r in jumps["row"]:
        before = touches[touches["row"] <= r]
        last = before.iloc[-1] if len(before) else None
        justified = last is not None and last["entity"] == entity and last["kind"] in ("Set", "Bump")
        if not justified and not serve_turn[r]:
            random_jumps += 1
        if last is not None and last["entity"] == entity and last["kind"] == "Set":
            after = touches[touches["row"] > r]
            if len(after) and after.iloc[0]["entity"] == entity and after.iloc[0]["kind"] in SPIKE_STATES:
                chains += 1

    # ---- mất điểm thế nào: ai chạm cuối, bằng cú gì
    causes = Counter()
    for r in lost_at:
        before = touches[touches["row"] < r]
        if not len(before):
            causes["(chưa ai chạm)"] += 1
            continue
        last = before.iloc[-1]
        who = "mình" if last["entity"] == entity else ("đối thủ" if last["entity"] == other[2] else "khác")
        causes[f"{who} chạm cuối bằng {last['kind']}"] += 1

    hits = Counter(mine["kind"])
    return {
        "points_won": won,
        "points_lost": int(len(lost_at)),
        "alive_frames": int(alive.sum()),
        "idle_frames": idle,
        "actions": int(len(onsets)),
        "attempts": dict(Counter(onsets["intent"])),
        "hits": dict(hits),
        "zone_episodes": zone_episodes,
        "zone_missed": missed,
        "jumps": int(len(jumps)),
        "random_jumps": random_jumps,
        "set_jump_spike": chains,
        "loss_causes": dict(causes),
        "longest_rally": max((e - s + 1 for s, e in rallies(df)), default=0),
    }


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict):
            out[k] = dict(Counter(a.get(k, {})) + Counter(v))
        elif k == "longest_rally":
            out[k] = max(a.get(k, 0), v)
        else:
            out[k] = a.get(k, 0) + v
    return out


def summarize(raw: dict, n_matches: int) -> dict:
    """Từ số đếm thô sang các chỉ số báo cáo (tỉ lệ, trung bình mỗi trận)."""
    hits, attempts = raw["hits"], raw["attempts"]
    spike_hits = sum(v for k, v in hits.items() if k in SPIKE_STATES)
    spike_attempts = sum(v for k, v in attempts.items() if k in SPIKE_INTENTS)
    all_hits = sum(hits.values())
    points = raw["points_won"] + raw["points_lost"]

    def per_match(v):
        return v / n_matches if n_matches else 0.0

    def pct(a, b):
        return 100.0 * a / b if b else 0.0

    return {
        "tran": n_matches,
        "diem_thang": raw["points_won"],
        "diem_mat": raw["points_lost"],
        "pct_diem_thang": pct(raw["points_won"], points),
        "diem_mat_moi_tran": per_match(raw["points_lost"]),
        "hanh_dong_moi_tran": per_match(raw["actions"]),
        "cu_dap_bam_moi_tran": per_match(spike_attempts),
        "cu_dap_trung_moi_tran": per_match(spike_hits),
        "ti_trong_tan_cong_pct": pct(spike_hits, all_hits),
        "giao_bong_moi_tran": per_match(hits.get("Serve", 0)),
        "ti_le_dung_yen_pct": pct(raw["idle_frames"], raw["alive_frames"]),
        "ti_le_bo_lo_bong_vao_tam_pct": pct(raw["zone_missed"], raw["zone_episodes"]),
        "set_moi_tran": per_match(hits.get("Set", 0)),
        "bump_moi_tran": per_match(hits.get("Bump", 0)),
        "jump_moi_tran": per_match(raw["jumps"]),
        "chuoi_set_jump_spike": raw["set_jump_spike"],
        "jump_bua": raw["random_jumps"],
        "phan_bo_cu_dap_pct": {k: pct(hits.get(k, 0), spike_hits) for k in SPIKE_KINDS},
        "cach_mat_diem": dict(sorted(raw["loss_causes"].items(), key=lambda kv: -kv[1])),
        "pha_bong_dai_nhat_frame": raw["longest_rally"],
    }


def evaluate(paths) -> dict:
    files = _collect(paths)
    if not files:
        raise FileNotFoundError(f"Không có file log nào trong {paths}")
    raw = {side: {} for side in SIDES}
    who = {side: Counter() for side in SIDES}
    per_file = []
    for f in files:
        df, _ = load_match(f)
        if df.empty:
            continue
        row = {"file": f.name}
        for side in SIDES:
            m = side_metrics(df, side)
            raw[side] = _merge(raw[side], m)
            ctrl = controller_of(df, SIDES[side][1])
            who[side][ctrl] += 1
            row[side] = {"ai": ctrl, "diem": m["points_won"]}
        per_file.append(row)
    n = len(per_file)
    return {
        "files": per_file,
        "sides": {s: {"ai": " / ".join(who[s]), **summarize(raw[s], n)} for s in SIDES},
    }


def _collect(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += [x for x in log_files(p) if not is_feature_dump(x)]
        elif p.name.endswith((".csv", ".csv.gz")):
            out.append(p)
    return out


LABELS = [
    ("pct_diem_thang", "% điểm thắng", "{:.1f}%"),
    ("diem_mat_moi_tran", "điểm mất / trận", "{:.1f}"),
    ("hanh_dong_moi_tran", "hành động / trận", "{:.1f}"),
    ("cu_dap_bam_moi_tran", "cú đập bấm / trận", "{:.1f}"),
    ("cu_dap_trung_moi_tran", "cú đập trúng / trận", "{:.1f}"),
    ("ti_trong_tan_cong_pct", "tỉ trọng tấn công", "{:.1f}%"),
    ("giao_bong_moi_tran", "giao bóng / trận", "{:.1f}"),
    ("ti_le_dung_yen_pct", "tỉ lệ đứng yên", "{:.1f}%"),
    ("ti_le_bo_lo_bong_vao_tam_pct", "bỏ lỡ bóng vào tầm", "{:.1f}%"),
    ("set_moi_tran", "Set / trận", "{:.1f}"),
    ("bump_moi_tran", "Bump / trận", "{:.1f}"),
    ("jump_moi_tran", "Jump / trận", "{:.1f}"),
    ("chuoi_set_jump_spike", "chuỗi Set->Jump->Spike", "{}"),
    ("jump_bua", "Jump bừa", "{}"),
    ("pha_bong_dai_nhat_frame", "pha bóng dài nhất (frame)", "{:,}"),
]


def print_report(report: dict) -> None:
    sides = report["sides"]
    left, right = sides["trái"], sides["phải"]
    score = f"trái ({left['ai']}) {left['diem_thang']} - {right['diem_thang']} phải ({right['ai']})"
    print(f"{left['tran']} trận. Điểm: {score}\n")
    print(f"{'chỉ số':<28}{'trái':>16}{'phải':>16}")
    print(f"{'':<28}{left['ai'][:15]:>16}{right['ai'][:15]:>16}")
    for key, label, fmt in LABELS:
        print(f"{label:<28}{fmt.format(left[key]):>16}{fmt.format(right[key]):>16}")
    for k in SPIKE_KINDS:
        a, b = left["phan_bo_cu_dap_pct"][k], right["phan_bo_cu_dap_pct"][k]
        print(f"{'  ' + k:<28}{a:>15.1f}%{b:>15.1f}%")
    for side in ("trái", "phải"):
        causes = sides[side]["cach_mat_diem"]
        if causes:
            print(f"\nSân {side} ({sides[side]['ai']}) mất điểm khi:")
            for c, v in causes.items():
                print(f"  {v:>4}  {c}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="file .csv hoặc thư mục log")
    parser.add_argument("--json", help="ghi kết quả ra file JSON")
    args = parser.parse_args()
    report = evaluate(args.paths)
    print_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
