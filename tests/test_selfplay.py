"""Test vong tu hoc: giu model tot hon, tra lai model cu khi model moi kem hon; khong dong toi game chinh."""

from pathlib import Path

import pytest
import yaml

from spike_ai import selfplay


def test_decide_needs_a_real_gain():
    assert selfplay.decide(40.0, 42.0, 0.0)
    assert not selfplay.decide(40.0, 40.0, 0.0)
    assert not selfplay.decide(40.0, 41.0, 2.0)


def test_train_config_drops_selfplay_section(tmp_path):
    cfg = {"run_name": "sp", "schema": "v1", "selfplay": {"game_dir": "x"}}
    path = selfplay.train_config(cfg, tmp_path, 3)
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "selfplay" not in written and written["run_name"] == "sp_iter03"


@pytest.fixture
def fake_game(tmp_path, monkeypatch):
    """Repo game gia: 2 header; MatchRunner, build, train, export deu gia lap."""
    game = tmp_path / "game"
    (game / "Source/AI/Generated").mkdir(parents=True)
    (game / "Source/AI/Generated" / selfplay.HEADER_NAME).write_text("model game chinh", encoding="utf-8")
    monkeypatch.setattr(selfplay, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(selfplay, "REPORTS_DIR", tmp_path)
    (tmp_path / "experiments.md").write_text("", encoding="utf-8")
    monkeypatch.setattr(selfplay, "build_runner", lambda sp: None)
    monkeypatch.setattr(selfplay, "run_matches", lambda sp, out, n, seed, tag: "TONG: gia")

    def fake_train(config_path, run_name):
        return Path(config_path).with_suffix(".joblib")

    def fake_export(model, out_dir):
        out = Path(out_dir) / selfplay.HEADER_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"model {Path(model).stem}", encoding="utf-8")
        return out

    monkeypatch.setattr(selfplay, "train_candidate", fake_train)
    monkeypatch.setattr(selfplay.export_cpp, "export", fake_export)
    cfg = {
        "run_name": "sp",
        "schema": "v1",
        "selfplay": {
            "game_dir": str(game),
            "vcvars": "x",
            "auto_logs": str(tmp_path / "auto"),
            "matches_per_iteration": 2,
        },
    }
    path = tmp_path / "sp.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return game, path


def _scores(monkeypatch, values):
    it = iter(values)
    monkeypatch.setattr(selfplay, "score_ml", lambda sp, d: {"pct_diem_thang": next(it)})


def test_better_model_is_kept_worse_model_is_rolled_back(fake_game, monkeypatch):
    game, cfg = fake_game
    _scores(monkeypatch, [40.0, 45.0, 43.0])  # xuat phat, vong 1 tot hon, vong 2 kem hon
    out = selfplay.run(str(cfg), iterations=2)

    kept = [h["kept"] for h in out["history"]]
    assert kept == [True, True, False]
    auto = (game / "Source/AI/GeneratedAuto" / selfplay.HEADER_NAME).read_text(encoding="utf-8")
    assert auto == "model train_iter01"  # vong 2 bi bo -> tra lai model vong 1
    best = Path(out["run_dir"]) / "best" / selfplay.HEADER_NAME
    assert best.read_text(encoding="utf-8") == "model train_iter01"


def test_main_game_model_is_never_touched(fake_game, monkeypatch):
    game, cfg = fake_game
    _scores(monkeypatch, [40.0, 50.0])
    selfplay.run(str(cfg), iterations=1)
    assert (game / "Source/AI/Generated" / selfplay.HEADER_NAME).read_text(encoding="utf-8") == "model game chinh"


def test_compressed_logs_are_read_like_plain_csv(tmp_path):
    """Log tự sinh được nén .csv.gz: bộ đọc dữ liệu và eval_ingame phải đọc như .csv thường."""
    import numpy as np
    import pandas as pd

    from spike_ai.data import load_dirs
    from spike_ai.schema import load_spec, raw_column_names

    folder = tmp_path / "auto" / "iter_01"
    folder.mkdir(parents=True)
    df = pd.DataFrame({c: np.zeros(5) for c in raw_column_names(load_spec("v1"))})
    df["schema_version"], df["match_id"] = 1, "auto_s1_m000_BOTRULE"
    df.to_csv(folder / "auto_s1_m000_BOTRULE.csv", index=False)
    (folder / "x_feat.csv").write_text("a\n1\n", encoding="utf-8")

    selfplay.compress_logs(folder)

    assert sorted(p.name for p in folder.iterdir()) == ["auto_s1_m000_BOTRULE.csv.gz", "x_feat.csv.gz"]
    loaded = load_dirs([tmp_path / "auto"], expected_version="v1")
    assert len(loaded) == 5 and set(loaded["match_id"]) == {"auto_s1_m000_BOTRULE"}


def test_eval_logs_are_deleted_after_scoring(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval" / "iter01"

    def fake_run(sp, out, n, seed, tag):
        out.mkdir(parents=True)
        (out / "m.csv").write_text("x", encoding="utf-8")
        return "TONG"

    fake_side = {
        "pct_diem_thang": 50.0,
        "diem_thang": 1,
        "diem_mat": 1,
        "cu_dap_trung_moi_tran": 1.0,
        "set_moi_tran": 0.0,
        "jump_moi_tran": 0.0,
        "chuoi_set_jump_spike": 0,
    }
    monkeypatch.setattr(selfplay, "run_matches", fake_run)
    monkeypatch.setattr(selfplay.eval_ingame, "evaluate", lambda paths: {"sides": {"phải": fake_side}})
    assert selfplay.score_ml({}, eval_dir)["pct_diem_thang"] == 50.0
    assert not eval_dir.exists()


def test_train_config_uses_only_the_last_selfplay_iterations(tmp_path):
    auto = tmp_path / "auto"
    for name in ("iter_A_01", "iter_A_02", "iter_B_01"):
        (auto / name).mkdir(parents=True)
    cfg = {
        "run_name": "sp",
        "data_dirs": [str(tmp_path / "v1"), str(auto)],
        "selfplay": {"auto_logs": str(auto), "keep_last_iterations": 2},
    }
    written = yaml.safe_load(selfplay.train_config(cfg, tmp_path, 1).read_text(encoding="utf-8"))
    assert [Path(d).name for d in written["data_dirs"]] == ["v1", "iter_A_02", "iter_B_01"]


def test_stops_after_consecutive_rejections(fake_game, monkeypatch):
    game, cfg = fake_game
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    data["selfplay"]["stop_after_rejections"] = 2
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")
    _scores(monkeypatch, [40.0, 39.0, 38.0, 99.0])  # 2 vong bi bo -> dung, khong chay vong 3
    out = selfplay.run(str(cfg), iterations=5)
    assert [h["iteration"] for h in out["history"]] == [0, 1, 2]
