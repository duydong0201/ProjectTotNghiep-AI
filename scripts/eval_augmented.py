import sys
from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spike_ai.data import load_dirs  # noqa: E402
from spike_ai.features import build  # noqa: E402

print("Đang nạp tập dữ liệu train và holdout...")
df_tr = load_dirs(["data/raw/v1/human"])
X_tr, y_tr = build(df_tr, "v1", "player")

df_ho = load_dirs(["data/raw/v1/human_holdout"])
X_ho, y_ho = build(df_ho, "v1", "player")

# 1. Đánh giá spike_choice
mask_tr = y_tr["spike_choice"].notna()
X_sp = X_tr[mask_tr]
y_sp = y_tr.loc[mask_tr, "spike_choice"]
players = df_tr.loc[mask_tr, "p_player_id"]

print("\n=== ĐÁNH GIÁ LEAVE-ONE-PLAYER-OUT (LOPO) TRÊN SPIKE_CHOICE ===")
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
for test_p in sorted(players.unique()):
    tr_idx = (players != test_p).values
    te_idx = (players == test_p).values
    dt.fit(X_sp[tr_idx], y_sp[tr_idx])
    preds = dt.predict(X_sp[te_idx])
    score = f1_score(y_sp[te_idx], preds, average="macro")
    print(f"  Train trên các người chơi khác, Test trên {test_p}: Macro-F1 = {score:.4f}")

# 2. Đánh giá Holdout P05
dt.fit(X_sp, y_sp)
mask_ho = y_ho["spike_choice"].notna()
X_sp_ho = X_ho[mask_ho]
y_sp_ho = y_ho.loc[mask_ho, "spike_choice"]
preds_ho = dt.predict(X_sp_ho)

print("\n=== ĐÁNH GIÁ TRÊN TẬP HOLDOUT HOÀN TOÀN MỚI (P05) ===")
print(f"  Holdout Macro-F1: {f1_score(y_sp_ho, preds_ho, average='macro'):.4f}")
print("\nBáo cáo phân loại chi tiết (Holdout):")
print(classification_report(y_sp_ho, preds_ho))

# 3. Đánh giá Move (Di chuyển)
print("\n=== ĐÁNH GIÁ MÔ HÌNH DI CHUYỂN (MOVE) ===")
dt_move = DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=42)
dt_move.fit(X_tr, y_tr["move"])
preds_move_ho = dt_move.predict(X_ho)
print(f"  Holdout Move Macro-F1: {f1_score(y_ho['move'], preds_move_ho, average='macro'):.4f}")
print(classification_report(y_ho["move"], preds_move_ho))
