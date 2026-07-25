"""
Unit tests for the windowed dataset builder.

These pin down the parts that are easy to get subtly wrong: how frames are sliced
into (overlapping) windows, that aggregates ignore missing (no-face) frames, and
that blink *rate* is derived from the running blink total.
"""

import pandas as pd
import pytest

from src.data.build_dataset import (
    aggregate_window,
    window_clip,
    window_starts,
)


def _frame(time_sec, **overrides):
    """A per-frame row with sane defaults; override only what a test cares about."""
    row = {
        "clip_id": "phone_001",
        "time_sec": time_sec,
        "face_count": 1,
        "gaze_ratio": 0.5,
        "gaze_direction": "CENTER",
        "ear": 0.3,
        "eyes_closed": 0.0,
        "blink_total": 0.0,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "person_count": 1,
        "phone_detected": 0.0,
        "phone_conf": 0.0,
        "book_detected": 0.0,
        "label_looking_away": 0,
        "label_phone": 1,
        "label_multiple_people": 0,
        "label_absent": 0,
    }
    row.update(overrides)
    return row


def _df(times, **per_frame_overrides):
    return pd.DataFrame([_frame(t, **per_frame_overrides) for t in times])


def test_window_starts_overlap():
    # 4s of frames, 2s window, 1s step -> starts at 0,1,2 (last full window [2,4]).
    assert window_starts(0.0, 4.0, window_sec=2.0, step_sec=1.0) == [0.0, 1.0, 2.0]


def test_clip_shorter_than_window_yields_one_window():
    assert window_starts(0.0, 1.2, window_sec=2.0, step_sec=1.0) == [0.0]


def test_windows_are_half_open_no_double_count():
    # A frame exactly on a window boundary belongs to the later window only.
    clip = _df([0.0, 1.0, 2.0, 3.0, 4.0])
    windows = window_clip(clip, window_sec=2.0, step_sec=2.0, min_frames=1)
    # Full windows only: [0,2) -> 0.0,1.0 ; [2,4) -> 2.0,3.0. The frame at 4.0
    # would need a [4,6) window that runs past the clip, so it's dropped.
    assert [w["n_frames"] for w in windows] == [2, 2]


def test_label_is_carried_onto_each_window():
    clip = _df([0.0, 0.5, 1.0])
    windows = window_clip(clip, window_sec=2.0, step_sec=1.0, min_frames=1)
    assert windows[0]["label_phone"] == 1
    assert windows[0]["clip_id"] == "phone_001"


def test_aggregates_ignore_missing_frames():
    # Two real gaze readings and one no-face frame (NaN) -> mean over the reals.
    win = pd.DataFrame([
        _frame(0.0, gaze_ratio=0.2),
        _frame(0.5, gaze_ratio=0.8),
        _frame(1.0, gaze_ratio=float("nan"), gaze_direction=None, face_count=0),
    ])
    result = aggregate_window(win, "phone_001", 0, 0.0, 2.0)
    assert result["gaze_ratio_mean"] == pytest.approx(0.5)    # NaN frame skipped
    assert result["frac_no_face"] == pytest.approx(1 / 3)     # 1 of 3 had no face


def test_blink_rate_from_running_total():
    # blink_total climbs 0 -> 2 across a 2s window => 1 blink/sec.
    win = pd.DataFrame([
        _frame(0.0, blink_total=0.0),
        _frame(1.0, blink_total=1.0),
        _frame(1.9, blink_total=2.0),
    ])
    result = aggregate_window(win, "phone_001", 0, 0.0, 2.0)
    assert result["blink_rate"] == pytest.approx(1.0)


def test_std_captures_movement():
    # A still head vs. a swinging one: std separates them even if means match.
    still = aggregate_window(_df([0.0, 0.5, 1.0], head_yaw=0.0), "c", 0, 0.0, 2.0)
    moving = aggregate_window(
        pd.DataFrame([
            _frame(0.0, head_yaw=-30.0),
            _frame(0.5, head_yaw=0.0),
            _frame(1.0, head_yaw=30.0),
        ]),
        "c", 0, 0.0, 2.0,
    )
    assert still["head_yaw_std"] == pytest.approx(0.0)
    assert moving["head_yaw_std"] > 10.0
