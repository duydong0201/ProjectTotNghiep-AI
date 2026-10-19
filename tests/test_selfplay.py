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
