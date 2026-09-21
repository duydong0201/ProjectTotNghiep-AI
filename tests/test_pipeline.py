"""Test cho các khâu quan trọng nhất: thứ tự feature, lật sân, chia theo trận, export C++."""

import numpy as np
import pandas as pd
import pytest

from spike_ai.export_cpp import predict_proba_from_tables, render_header, tree_tables
from spike_ai.features import build
from spike_ai.models import create
from spike_ai.schema import detect_version, feature_names, load_spec, raw_column_names
from spike_ai.split import train_test_by_match

NET_X = 1370.0


def make_v0(n=40):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Frame": np.arange(n),
            "PlayerPosX": rng.uniform(100, 1300, n),
            "PlayerPosY": 265.0,
            "PlayerEvent": rng.choice(["None", "MoveLeft", "MoveRight", "Bump"], n),
            "BotPosX": rng.uniform(1450, 2600, n),
            "BotPosY": 265.0,
            "BotEvent": rng.choice(["None", "MoveLeft", "MoveRight", "SpikeLight"], n),
            "match_id": np.repeat([f"m{i}" for i in range(4)], n // 4),
        }
    )


def make_v1(n=8):
    spec = load_spec("v1")
    df = pd.DataFrame({c: np.zeros(n) for c in raw_column_names(spec)})
    df["schema_version"] = 1
    df["match_id"] = "m0"
    df["p_x"], df["o_x"] = 500.0, 2000.0
    df["p_y"] = df["o_y"] = 265.0
    df["ball_x"], df["ball_y"] = 1800.0, 600.0
    df["ball_speed"] = 10.0              # bóng bay sang phải
    df["ball_a"], df["ball_b"] = -0.01, -1900.0   # đỉnh parabol tại x = 1900
    df["ball_landing_x"] = [2100.0] * (n - 1) + [9999.0]
    df["ball_state_frame"] = -1
    df["rally_last_touch"] = 0
    df["serving_team"] = 1               # RIGHT
    df["score_left"], df["score_right"] = 3, 5
    df["o_input_move"] = -7.0            # opponent chạy sang trái = về phía lưới
    df["o_input_intent"] = 9             # SpikeLight
    return df


# ---------------------------------------------------------------- schema
def test_detect_version():
    assert detect_version(make_v0().columns[:7]) == "v0"
    assert detect_version(make_v1().columns) == "v1"
    with pytest.raises(ValueError):
        detect_version(["a", "b"])


@pytest.mark.parametrize("version,maker", [("v0", make_v0), ("v1", make_v1)])
@pytest.mark.parametrize("agent", ["player", "opponent"])
def test_feature_order_matches_spec(version, maker, agent):
    X, y = build(maker(), version, agent)
    assert list(X.columns) == feature_names(load_spec(version))
    assert set(y.columns) == {"move", "action"}
    assert not X.isna().any().any()


# ---------------------------------------------------------------- lật sân
def test_v0_mirror_is_symmetric():
    """Hai nhân vật đứng đối xứng qua lưới phải cho cùng feature từ góc nhìn của mỗi bên."""
    df = pd.DataFrame(
        {"Frame": [0], "PlayerPosX": [NET_X - 300], "PlayerPosY": [265.0], "PlayerEvent": ["MoveRight"],
         "BotPosX": [NET_X + 300], "BotPosY": [265.0], "BotEvent": ["MoveLeft"], "match_id": ["m"]}
    )
    Xp, yp = build(df, "v0", "player")
    Xo, yo = build(df, "v0", "opponent")
    pd.testing.assert_frame_equal(Xp, Xo)
    # Cả hai đều tiến về lưới -> move = +1 trong hệ đã lật
    assert yp["move"].iloc[0] == 1 and yo["move"].iloc[0] == 1


def test_v1_opponent_view():
    X, y = build(make_v1(), "v1", "opponent")
    row = X.iloc[0]
    assert row["self_x"] == pytest.approx(2 * NET_X - 2000)
    assert row["ball_dx"] == pytest.approx((2 * NET_X - 1800) - (2 * NET_X - 2000))
    assert row["ball_vx"] == pytest.approx(-10.0)            # bóng bay về phía mình
    assert row["ball_vertex_dx"] == pytest.approx((2 * NET_X - 1900) - (2 * NET_X - 2000))
    assert row["landing_on_my_side"] == 1.0
    assert row["my_serve"] == 1.0
    assert row["score_diff"] == 2.0
    assert X.iloc[-1]["landing_valid"] == 0.0 and X.iloc[-1]["landing_dx"] == 0.0
    assert y["move"].iloc[0] == 1 and y["action"].iloc[0] == "SpikeLight"


# ---------------------------------------------------------------- split
def test_split_never_shares_matches():
    df = make_v0(80)
    train_idx, test_idx = train_test_by_match(df["match_id"], test_size=0.25, seed=1)
    assert set(df["match_id"].iloc[train_idx]).isdisjoint(df["match_id"].iloc[test_idx])


# ---------------------------------------------------------------- export
@pytest.mark.parametrize("name,params", [("decision_tree", {"max_depth": 4}),
                                         ("random_forest", {"n_estimators": 5, "max_depth": 4})])
def test_export_tables_match_sklearn(name, params):
    X, y = build(make_v0(200), "v0", "opponent")
    model = create(name, seed=0, params=params).fit(X, y["move"])
    tables = [tree_tables(t) for t in getattr(model, "estimators_", [model])]
    for x, expected in zip(X.to_numpy(), model.predict_proba(X)):
        np.testing.assert_allclose(predict_proba_from_tables(tables, x), expected, atol=1e-6)

    bundle = {"target": "move", "model_name": name, "schema": "v0", "agent": "opponent",
              "features": list(X.columns), "classes": [str(c) for c in model.classes_],
              "golden_X": X.head(3).to_numpy().tolist(), "golden_proba": model.predict_proba(X.head(3)).tolist()}
    header = render_header(bundle, tables)
    assert "RunGoldenTest" in header and f"kNumTrees    = {len(tables)}" in header
