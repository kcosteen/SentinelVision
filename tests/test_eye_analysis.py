"""
Unit tests for the Eye Aspect Ratio (EAR) calculation used in blink detection.

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

It measures how "open" an eye is: a wide-open eye has a larger vertical gap
(numerator) relative to the eye width (denominator), so a *high* EAR; during a
blink the eyelids close, the vertical gap collapses, and EAR drops sharply.

To test the math in isolation we feed in fake landmarks with known pixel
coordinates and set width = height = 1 so the coordinates pass through
unchanged (no floating-point rounding to reason about).
"""

import pytest

from src.features.eye_analysis import calculate_ear


class FakeLandmark:
    """Stand-in for a MediaPipe landmark, which only needs `.x` and `.y`."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_calculate_ear_known_value():
    # Points laid out so we can compute EAR by hand:
    #   p1=(0,50)  p4=(60,50)  -> horizontal width = 60
    #   p2=(20,40) p6=(20,60)  -> vertical gap = 20
    #   p3=(40,40) p5=(40,60)  -> vertical gap = 20
    # EAR = (20 + 20) / (2 * 60) = 40 / 120 = 1/3
    landmarks = [
        FakeLandmark(0, 50),
        FakeLandmark(20, 40),
        FakeLandmark(40, 40),
        FakeLandmark(60, 50),
        FakeLandmark(40, 60),
        FakeLandmark(20, 60),
    ]
    eye_points = [0, 1, 2, 3, 4, 5]

    ear = calculate_ear(eye_points, landmarks, width=1, height=1)

    assert ear == pytest.approx(1 / 3)


def test_open_eye_has_higher_ear_than_blinking_eye():
    eye_points = [0, 1, 2, 3, 4, 5]

    # Wide-open eye: large vertical gaps.
    open_eye = [
        FakeLandmark(0, 50),
        FakeLandmark(20, 30),
        FakeLandmark(40, 30),
        FakeLandmark(60, 50),
        FakeLandmark(40, 70),
        FakeLandmark(20, 70),
    ]
    # Blinking eye: eyelids nearly touching (tiny vertical gaps).
    blinking_eye = [
        FakeLandmark(0, 50),
        FakeLandmark(20, 49),
        FakeLandmark(40, 49),
        FakeLandmark(60, 50),
        FakeLandmark(40, 51),
        FakeLandmark(20, 51),
    ]

    ear_open = calculate_ear(eye_points, open_eye, width=1, height=1)
    ear_blink = calculate_ear(eye_points, blinking_eye, width=1, height=1)

    assert ear_open > ear_blink
