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
