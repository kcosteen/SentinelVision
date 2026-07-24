"""
Unit tests for multi-label parsing.

Proctoring behaviors co-occur, so a clip can carry several labels at once. These
tests pin down that `parse_labels` handles one label, several labels, the
"normal" / empty cases, and rejects typos loudly.
"""

import pytest

from src.data.labels import BEHAVIOR_LABELS, LABEL_COLUMNS, parse_labels


def test_single_label():
    result = parse_labels("phone")
    assert result["label_phone"] == 1
    # everything else is 0
    assert sum(result.values()) == 1


def test_multiple_co_occurring_labels():
    result = parse_labels("looking_away,phone")
    assert result["label_looking_away"] == 1
    assert result["label_phone"] == 1
    assert result["label_multiple_people"] == 0
    assert result["label_absent"] == 0


def test_normal_and_empty_mean_all_zeros():
    assert parse_labels("normal") == {col: 0 for col in LABEL_COLUMNS}
    assert parse_labels("") == {col: 0 for col in LABEL_COLUMNS}
    assert parse_labels(None) == {col: 0 for col in LABEL_COLUMNS}


def test_whitespace_is_tolerated():
    result = parse_labels(" looking_away , phone ")
    assert result["label_looking_away"] == 1
    assert result["label_phone"] == 1


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        parse_labels("cheating")


def test_columns_match_behaviors():
    assert LABEL_COLUMNS == [f"label_{name}" for name in BEHAVIOR_LABELS]
