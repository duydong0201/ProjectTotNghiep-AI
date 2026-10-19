"""Kiểm tra và xuất báo cáo đối soát 5 trận DAgger vừa sinh theo đúng các tiêu chí yêu cầu."""

from pathlib import Path

import pandas as pd

log_dir = Path(__file__).resolve().parents[1].parent / "ProjectTotNghiep" / "Logs" / "v1"
files = [
    "2026-10-15_23-01-14_BOTRULE.csv",
    "2026-10-15_23-08-35_BOTRULE.csv",
    "2026-10-15_23-16-02_BOTRULE.csv",
    "2026-10-15_23-23-45_BOTRULE.csv",
    "2026-10-15_23-31-10_BOTRULE.csv",
]

report_rows = []

for fname in files:
    fpath = log_dir / fname
    df = pd.read_csv(fpath)
    h = df[df["o_controller"] == 0]
    counts = h["o_input_intent"].value_counts().to_dict()

    jump_serves = 0
    serve_idx = df[(df["o_input_intent"] == 8) & (df["o_controller"] == 0)].index
    for idx in serve_idx:
        prev = df.loc[max(0, idx - 25) : idx, "o_input_intent"]
        if (prev == 1).any():
            jump_serves += 1

    serve1_df = df[(df["serving_team"] == 1) & (df["ball_state_frame"] > 0)]
    st1_takeover = (serve1_df["o_controller"] == 0).mean() if len(serve1_df) > 0 else 0.0
    bump_touch_cnt = (df["rally_last_touch_kind"] == 5).sum()
    sl = df["score_left"].iloc[-1]
    sr = df["score_right"].iloc[-1]

    row = {
        "File": fname,
        "Rows": len(df),
        "Score": f"{sl} - {sr}",
        "p_ctrl==1": (df["p_controller"] == 1).all(),
        "NaNs": int(df.isna().sum().sum()),
        "o_ctrl==0 (%)": f"{len(h) / len(df):.1%}",
        "move!=0 (%)": f"{(h['o_input_move'] != 0).mean():.1%}",
        "Jump (1)": counts.get(1, 0),
        "Set (7)": counts.get(7, 0),
        "Bump (5)": counts.get(5, 0),
        "SpikeLight (9)": counts.get(9, 0),
        "SpikeMed (10)": counts.get(10, 0),
        "SpikeStrong (11)": counts.get(11, 0),
        "Serve (8)": counts.get(8, 0),
        "JumpServes": jump_serves,
        "TouchKind==5": bump_touch_cnt,
        "Serve1 Takeover": f"{st1_takeover:.1%}",
    }
    report_rows.append(row)

rep_df = pd.DataFrame(report_rows)
print(rep_df.to_string(index=False))
