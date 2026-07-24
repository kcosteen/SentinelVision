"""
Unit tests for gaze estimation.

`estimate_gaze` decides whether the person is looking LEFT / RIGHT / CENTER by
computing where the iris sits between the two eye corners:

    ratio = (iris_x - left_corner_x) / (right_corner_x - left_corner_x)

ratio ~0.0 -> iris near the left corner, ~1.0 -> near the right corner.
The thresholds are: ratio < 0.35 -> LEFT, ratio > 0.65 -> RIGHT, else CENTER.

We build fake landmarks only for the indices the function actually reads
(the left-eye corners and the 4 left-iris points), using width = height = 1 so
coordinates pass through unchanged.
"""

import pytest

from src.features.gaze_estimation import estimate_gaze, gaze_ratio


class FakeLandmark:
    def __init__(self, x, y=0):
        self.x = x
        self.y = y


def make_landmarks(iris_x):
    """Left corner at x=20, right corner at x=60 (eye width = 40).

    The 4 iris points all share the same x so the iris centre = iris_x.
    """
    return {
        362: FakeLandmark(20),   # LEFT_EYE_CORNERS[0]
        263: FakeLandmark(60),   # LEFT_EYE_CORNERS[1]
        474: FakeLandmark(iris_x),  # LEFT_IRIS ...
        475: FakeLandmark(iris_x),
        476: FakeLandmark(iris_x),
        477: FakeLandmark(iris_x),
    }


def test_gaze_center():
    # iris at x=40 -> ratio = (40-20)/40 = 0.5 -> CENTER
    assert estimate_gaze(make_landmarks(40), width=1, height=1) == "CENTER"


def test_gaze_left():
    # iris at x=25 -> ratio = (25-20)/40 = 0.125 (< 0.35) -> LEFT
    assert estimate_gaze(make_landmarks(25), width=1, height=1) == "LEFT"


def test_gaze_right():
    # iris at x=55 -> ratio = (55-20)/40 = 0.875 (> 0.65) -> RIGHT
    assert estimate_gaze(make_landmarks(55), width=1, height=1) == "RIGHT"


def test_gaze_ratio_returns_continuous_value():
    # The raw ratio (used as an ML feature) is the number the LEFT/RIGHT/CENTER
    # label is derived from: (iris_x - left_corner) / eye_width.
    assert gaze_ratio(make_landmarks(40), width=1, height=1) == pytest.approx(0.5)
    assert gaze_ratio(make_landmarks(25), width=1, height=1) == pytest.approx(0.125)
    assert gaze_ratio(make_landmarks(55), width=1, height=1) == pytest.approx(0.875)
