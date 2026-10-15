"""Test cho bước dựng lại frame của log v0 (data.rebuild_frames_v0)."""

import pandas as pd
import pytest

from spike_ai.data import MAX_FILL_GAP, rebuild_frames_v0
from spike_ai.features import build
from spike_ai.schema import load_spec, raw_column_names

RAW_COLUMNS = raw_column_names(load_spec("v0"))


def raw(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_merges_two_rows_of_the_same_frame():
    """Cả hai cùng di chuyển -> game ghi hai dòng, mỗi dòng ghi sai bên kia thành None.

    Số liệu lấy nguyên từ Logs/2026-08-06_15-46-15.csv frame 416.
    """
    out = rebuild_frames_v0(
        raw(
            [
                (416, 596, 265, "MoveRight", 1801, 265, "None"),
                (416, 596, 265, "None", 1794, 265, "MoveLeft"),
            ]
        )
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["PlayerEvent"] == "MoveRight"
    assert row["BotEvent"] == "MoveLeft"
    # vị trí lấy ở dòng cuối - lúc cả hai đã di chuyển xong
    assert row["BotPosX"] == 1794


def test_both_agents_get_correct_move_label_after_rebuild():
    """Nhãn move của cả hai nhân vật phải đúng sau khi gộp (trước khi sửa: một bên bị 0)."""
    out = rebuild_frames_v0(
        raw(
            [
                (416, 596, 265, "MoveRight", 1801, 265, "None"),
                (416, 596, 265, "None", 1794, 265, "MoveLeft"),
            ]
        )
    )
    out["match_id"] = "m"

    _, y_player = build(out, "v0", "player")
    _, y_opponent = build(out, "v0", "opponent")
    # cả hai đều tiến về phía lưới -> +1 trong hệ toạ độ đã lật
    assert y_player["move"].iloc[0] == 1
    assert y_opponent["move"].iloc[0] == 1


def player_moves_right(frame: int, x: float) -> tuple:
    return (frame, x, 265, "MoveRight", 1800, 265, "None")


def test_fills_short_gap_as_standing_still():
    out = rebuild_frames_v0(raw([player_moves_right(10, 500), player_moves_right(14, 507)]))

    assert list(out["Frame"]) == [10, 11, 12, 13, 14]
    gap = out[out["Frame"].between(11, 13)]
    assert (gap["PlayerEvent"] == "None").all()
    assert (gap["BotEvent"] == "None").all()
    # vị trí giữ nguyên theo frame trước đó
    assert (gap["PlayerPosX"] == 500).all()


def test_keeps_long_gap_unfilled():
    """Lỗ hổng dài là lúc reset điểm - điền vào sẽ tạo ra nhãn đứng yên giả."""
    far = 10 + (MAX_FILL_GAP + 1) + 1  # cách nhau MAX_FILL_GAP + 1 frame trống
    out = rebuild_frames_v0(raw([player_moves_right(10, 500), player_moves_right(far, 900)]))

    assert list(out["Frame"]) == [10, far]


def test_fills_gap_exactly_at_the_limit():
    last = 10 + MAX_FILL_GAP + 1  # cách nhau đúng MAX_FILL_GAP frame trống
    out = rebuild_frames_v0(raw([player_moves_right(10, 500), player_moves_right(last, 900)]))

    assert list(out["Frame"]) == list(range(10, last + 1))


def test_output_has_one_row_per_frame_sorted():
    out = rebuild_frames_v0(
        raw(
            [
                (7, 500, 265, "None", 1800, 265, "MoveLeft"),
                (5, 480, 265, "MoveRight", 1810, 265, "None"),
                (5, 480, 265, "None", 1803, 265, "MoveLeft"),
                (5, 480, 500, "Spike", 1803, 265, "None"),
            ]
        )
    )

    assert list(out["Frame"]) == [5, 6, 7]
    assert not out["Frame"].duplicated().any()
    assert list(out.columns) == RAW_COLUMNS


def test_event_wins_over_none_regardless_of_row_order():
    """Dòng sự kiện bóng (LogTrajectoryEvent) đứng trước dòng di chuyển trong cùng frame."""
    out = rebuild_frames_v0(raw([(9, 600, 500, "Spike", 1800, 265, "None"), (9, 600, 520, "None", 1800, 265, "None")]))

    assert out["PlayerEvent"].iloc[0] == "Spike"
    assert out["PlayerPosY"].iloc[0] == 520


def test_ignores_rows_written_half_way():
    """File bị cắt giữa chừng cho ra event lạ / thiếu vị trí - không được làm hỏng cả trận."""
    out = rebuild_frames_v0(
        raw(
            [
                (3, 500, 265, "MoveRight", 1800, 265, "None"),
                (4, 507, 265, "Non", 1800, 265, "None"),
                (5, None, None, "MoveRight", None, None, "None"),
            ]
        )
    )

    assert list(out["Frame"]) == [3, 4]
    assert out["PlayerEvent"].iloc[1] == "None"


@pytest.mark.parametrize("rows", [[], [(1, 500, 265, "MoveLeft", 1800, 265, "None")]])
def test_handles_empty_and_single_row(rows):
    out = rebuild_frames_v0(raw(rows))
    assert len(out) == len(rows)


# ------------------------------------------------- suy ra rally_last_touch_kind (v1 -> v2)
def _kind_frames(rows):
    """rows = [(rally_last_touch, rally_touch_count, p_input_intent, o_input_intent), ...]"""
    import pandas as pd

    return pd.DataFrame(rows, columns=["rally_last_touch", "rally_touch_count", "p_input_intent", "o_input_intent"])


def test_rebuild_kind_maps_final_intent_to_action_state():
    """Hai enum KHAC nhau: Set la 7 ben FinalIntent nhung 6 ben ActionState."""
    from spike_ai.data import rebuild_kind_v1_to_v2

    df = _kind_frames([(0, 1, 7, 0), (0, 1, 0, 0), (0, 1, 0, 0)])  # player Set
    assert rebuild_kind_v1_to_v2(df).tolist() == [6, 6, 6]


def test_rebuild_kind_handles_first_row_already_touched():
    """Dong dau file thuong DA co san mot cu cham (giao bong xay ra truoc dong log dau).
    Thieu nhanh k == 0 thi mat ca pha bong dau - do duoc 73/5385 frame sai."""
    from spike_ai.data import rebuild_kind_v1_to_v2

    df = _kind_frames([(0, 1, 8, 0), (0, 1, 0, 0)])  # Serve ngay frame dau
    assert rebuild_kind_v1_to_v2(df).tolist() == [4, 4]


def test_rebuild_kind_resets_when_rally_resets():
    """RallyState.Reset() dat lastTouch = GameConfig::NONE = 8 (KHONG phai -1)
    va lastTouchKind = None."""
    from spike_ai.data import rebuild_kind_v1_to_v2

    df = _kind_frames([(0, 1, 7, 0), (8, 0, 0, 0), (8, 0, 0, 0)])
    assert rebuild_kind_v1_to_v2(df).tolist() == [6, 0, 0]


def test_rebuild_kind_detects_same_character_touching_twice():
    """Cung mot nhan vat cham hai lan: lastTouch KHONG doi, chi touchCount tang."""
    from spike_ai.data import rebuild_kind_v1_to_v2

    df = _kind_frames([(0, 1, 7, 0), (0, 2, 10, 0), (0, 2, 0, 0)])  # Set roi SpikeMedium
    assert rebuild_kind_v1_to_v2(df).tolist() == [6, 8, 8]


def test_rebuild_kind_ignores_jump_because_it_does_not_touch_the_ball():
    """Jump do JumpStartSystem xu ly, khong qua UpdateRallyTouch, nen khong bao gio
    duoc ghi vao lastTouchKind."""
    from spike_ai.data import rebuild_kind_v1_to_v2

    df = _kind_frames([(0, 1, 7, 0), (0, 1, 1, 0), (0, 1, 0, 0)])  # Set roi Jump
    assert rebuild_kind_v1_to_v2(df).tolist() == [6, 6, 6]


def test_rebuild_kind_reads_the_intent_of_the_entity_that_touched():
    """lastTouch = 3 (OPPONENT_1) thi phai doc o_input_intent, khong phai p_input_intent."""
    from spike_ai.data import rebuild_kind_v1_to_v2

    df = _kind_frames([(3, 1, 7, 10), (3, 1, 0, 0)])  # player de xuat Set, opponent dap
    assert rebuild_kind_v1_to_v2(df).tolist() == [8, 8]


# ------------------------------------------------- thu muc tron v1 + v2
def _v1_log(n=6):
    """Mot log v1 toi thieu, du cot theo spec."""
    import numpy as np
    import pandas as pd

    from spike_ai.schema import load_spec, raw_column_names

    df = pd.DataFrame({c: np.zeros(n) for c in raw_column_names(load_spec("v1"))})
    df["schema_version"] = 1
    df["match_id"] = "m0"
    df["frame"] = np.arange(n)
    df["rally_last_touch"] = 0
    df["rally_touch_count"] = 1
    df["p_input_intent"] = [7] + [0] * (n - 1)  # FinalIntent.Set
    return df


def test_adapt_v1_to_v2_adds_the_derived_column():
    from spike_ai.data import adapt_version

    out = adapt_version(_v1_log(), "v1", "v2")
    assert "rally_last_touch_kind" in out.columns
    assert set(out["schema_version"].unique()) == {2}
    assert (out["rally_last_touch_kind"] == 6).all()  # ActionState.Set


def test_adapt_v2_to_v1_drops_the_extra_column():
    from spike_ai.data import adapt_version

    v2 = adapt_version(_v1_log(), "v1", "v2")
    back = adapt_version(v2, "v2", "v1")
    assert "rally_last_touch_kind" not in back.columns
    assert set(back["schema_version"].unique()) == {1}


def test_adapt_leaves_matching_version_untouched():
    import pandas as pd

    from spike_ai.data import adapt_version

    df = _v1_log()
    pd.testing.assert_frame_equal(adapt_version(df, "v1", "v1"), df)


def test_adapt_refuses_v0():
    import pytest

    from spike_ai.data import adapt_version

    with pytest.raises(ValueError, match="Không chuyển được schema"):
        adapt_version(_v1_log(), "v0", "v2")


def test_load_dirs_reads_a_folder_with_both_v1_and_v2(tmp_path):
    """Logs/v1 chua ca log ghi truoc 12/10/2026 (v1) va sau do (v2)."""
    from spike_ai.data import adapt_version, load_dirs

    v1 = _v1_log()
    v1.assign(match_id="m_v1").to_csv(tmp_path / "a_v1.csv", index=False)
    adapt_version(v1, "v1", "v2").assign(match_id="m_v2").to_csv(tmp_path / "b_v2.csv", index=False)

    as_v1 = load_dirs([tmp_path], expected_version="v1")
    assert "rally_last_touch_kind" not in as_v1.columns
    assert set(as_v1["schema_version"].unique()) == {1}
    assert as_v1["match_id"].nunique() == 2

    as_v2 = load_dirs([tmp_path], expected_version="v2")
    assert set(as_v2["schema_version"].unique()) == {2}
    assert as_v2["match_id"].nunique() == 2


def test_load_dirs_skips_feature_dump_files(tmp_path):
    """<match_id>_feat.csv la file doi chieu feature, khong phai log tran."""
    import pandas as pd

    from spike_ai.data import load_dirs

    _v1_log().to_csv(tmp_path / "match.csv", index=False)
    pd.DataFrame({"frame": [0, 1], "agent": [0, 3], "f0": [1.0, 2.0]}).to_csv(tmp_path / "match_feat.csv", index=False)
    df = load_dirs([tmp_path], expected_version="v1")
    assert "agent" not in df.columns
