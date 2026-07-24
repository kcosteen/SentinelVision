"""
Unit tests for the geometry helpers.

`distance` is a pure function (no camera, no ML), which makes it a perfect
place to start with testing: given known inputs we know the exact expected
output, so the test is deterministic.
"""

import math

from src.utils.geometry import distance


def test_distance_3_4_5_triangle():
    # Classic 3-4-5 right triangle -> hypotenuse of length 5.
    assert distance((0, 0), (3, 4)) == 5.0


def test_distance_is_zero_for_same_point():
    assert distance((7, 2), (7, 2)) == 0.0


def test_distance_is_symmetric():
    # distance(a, b) should equal distance(b, a).
    a, b = (1, 2), (4, 6)
    assert distance(a, b) == distance(b, a)


def test_distance_matches_manual_calculation():
    expected = math.sqrt((5 - 1) ** 2 + (1 - (-2)) ** 2)  # sqrt(16 + 9) = 5
    assert distance((1, -2), (5, 1)) == expected
