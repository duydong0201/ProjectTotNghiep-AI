"""Do phan bo hanh dong cua bot ML trong log bot-vs-bot.

Dung de tra loi: viec bo argmax (lay mau theo xac suat) co pha duoc tinh trang
97,1% cu dap la SpikeMedium hay khong.

Bo cuc san trong che do bot-vs-bot (MatchRuleConfig::ENABLE_BOT_VS_BOT = true):
  - san TRAI  = bot LUAT        -> cot p_input_*
  - san PHAI  = bot ML          -> cot o_input_*   (ML_AGENT_ENTITY = 3)

LUU Y khi doc ket qua: diem so o che do nay KHONG so sanh truc tiep duoc voi bang
nguoi-vs-bot trong BAO-CAO-TIEN-DO-2.md, vi doi thu khac nhau. Chi so so sanh duoc
la PHAN BO LOAI CU DAP cua chinh bot ML.

Cach dung:
    python scripts/analyze_botvsbot.py [thu_muc_log]
Mac dinh doc F:/DoAnTotNghiep/ProjectTotNghiep/Logs/v1_botvsbot
"""

import collections
import glob
import os
import sys

import pandas as pd

DEFAULT_DIR = "F:/DoAnTotNghiep/ProjectTotNghiep/Logs/v1_botvsbot"

# Thu tu phai khop enum class FinalIntent trong Source/Gameplay/FinalIntent.h
INTENT_NAMES = [
    "None",
    "Jump",
    "Spike",
    "Feint",
    "Block",
    "Bump",
    "Slide",
    "Set",
    "Serve",
    "SpikeLight",
    "SpikeMedium",
    "SpikeStrong",
]
SPIKES = ("SpikeLight", "SpikeMedium", "SpikeStrong")

# Moc do duoc tu 5 tran nguoi-vs-bot ML ngay 09/10/2026, khi con dung argmax.
BASELINE_ARGMAX = {"SpikeMedium": 407, "SpikeLight": 6, "SpikeStrong": 6}


def intent_name(value) -> str:
    try:
        i = int(value)
    except (TypeError, ValueError):
        return str(value)
    return INTENT_NAMES[i] if 0 <= i < len(INTENT_NAMES) else str(i)


def count_intents(series) -> collections.Counter:
    out = collections.Counter()
    for value, n in series.fillna(0).value_counts().items():
        out[intent_name(value)] += int(n)
    return out


def describe(title: str, intents: collections.Counter, moves: collections.Counter, frames: int) -> None:
    actions = {k: v for k, v in intents.items() if k != "None"}
    total = sum(actions.values())
    print(f"\n--- {title} ---")
    print(f"  tong hanh dong : {total}")
    if not total:
        print("  (khong co hanh dong nao)")
        return
    for name, n in sorted(actions.items(), key=lambda kv: -kv[1]):
        print(f"      {name:<13}{n:>6}  ({n / total * 100:5.1f}%)")
    standstill = moves.get(0, 0)
    print(f"  dung yen       : {standstill / frames * 100:.1f}% frame")


def main() -> int:
    log_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    files = [f for f in files if not f.endswith("_feat.csv")]
    if not files:
        print(f"Khong tim thay file .csv nao trong {log_dir}")
        print("Da chay game o che do ENABLE_BOT_VS_BOT = true chua?")
        return 1

    ml_intents, ml_moves = collections.Counter(), collections.Counter()
    rule_intents, rule_moves = collections.Counter(), collections.Counter()
    frames = 0
    score_rule = score_ml = 0

    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        frames += len(df)
        # san phai = bot ML, san trai = bot luat
        ml_intents += count_intents(df["o_input_intent"])
        rule_intents += count_intents(df["p_input_intent"])
        for col, dest in (("o_input_move", ml_moves), ("p_input_move", rule_moves)):
            for value, n in df[col].fillna(0).value_counts().items():
                dest[int(value)] += int(n)
        score_rule += int(df["score_left"].max())
        score_ml += int(df["score_right"].max())

    print("=" * 72)
    print(f"{len(files)} tran bot-vs-bot, {frames} frame")
    print(f"Diem: bot LUAT (trai) {score_rule}  -  bot ML (phai) {score_ml}")
    if score_rule + score_ml:
        print(f"      bot ML thang {score_ml / (score_rule + score_ml) * 100:.0f}% diem")

    describe("BOT ML (san phai, co lay mau theo xac suat)", ml_intents, ml_moves, frames)
    describe("BOT LUAT (san trai, moc so sanh)", rule_intents, rule_moves, frames)

    # Cau hoi chinh: con bi dinh vao SpikeMedium nua khong?
    now = {s: ml_intents.get(s, 0) for s in SPIKES}
    total_now = sum(now.values())
    total_old = sum(BASELINE_ARGMAX.values())
    print("\n" + "=" * 72)
    print("PHAN BO CU DAP CUA BOT ML - truoc va sau khi bo argmax")
    print(f"{'cu dap':<14}{'argmax (cu)':>16}{'lay mau (moi)':>18}")
    for s in SPIKES:
        old = BASELINE_ARGMAX.get(s, 0)
        old_pct = f"{old / total_old * 100:.1f}%" if total_old else "-"
        new_pct = f"{now[s] / total_now * 100:.1f}%" if total_now else "-"
        print(f"{s:<14}{old:>7} {old_pct:>8}{now[s]:>9} {new_pct:>8}")
    print(f"{'TONG':<14}{total_old:>7} {'100%':>8}{total_now:>9} {'100%':>8}")

    if total_now:
        top = max(now.values()) / total_now * 100
        print(f"\nCu dap ap dao nhat chiem {top:.1f}% (truoc khi sua: 97,1%)")
        print("=> SUA DUOC" if top < 80 else "=> VAN CON DON DIEU, can xem lai")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
