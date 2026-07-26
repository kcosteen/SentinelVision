"""
Unit tests for the threshold sweep.

Same principle as tests/test_metrics.py and tests/test_detection_metrics.py: this
code decides what constants go into the pipeline, so a bug here silently mis-tunes
the whole system. Every case below is hand-checkable.
"""

import pytest

from src.calibration.sweep import (
    best_threshold,
    candidate_thresholds,
    sweep_threshold,
)


def test_candidate_thresholds_endpoints_are_exact():
    """Float accumulation must not produce 0.30000000000000004."""
    values = candidate_thresholds(0.0, 1.0, 0.1)
    assert values[0] == 0.0
    assert values[-1] == 1.0
    assert 0.3 in values


def test_candidate_thresholds_is_inclusive_of_stop():
    assert candidate_thresholds(0.05, 0.15, 0.05) == [0.05, 0.10, 0.15]


def test_sweep_predicts_positive_at_or_above_threshold():
    """A score exactly equal to the threshold counts as positive."""
    rows = sweep_threshold([0.5], [1], [0.5])
    assert rows[0]["tp"] == 1 and rows[0]["fn"] == 0


def test_sweep_separates_perfectly_when_classes_do_not_overlap():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    rows = sweep_threshold(scores, labels, [0.5])
    assert rows[0]["precision"] == 1.0
    assert rows[0]["recall"] == 1.0
    assert rows[0]["f1"] == 1.0


def test_sweep_at_zero_threshold_predicts_everything_positive():
    rows = sweep_threshold([0.1, 0.9], [0, 1], [0.0])
    assert rows[0]["tp"] == 1 and rows[0]["fp"] == 1 and rows[0]["fn"] == 0
    assert rows[0]["recall"] == 1.0


def test_sweep_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        sweep_threshold([0.1, 0.2], [1])


def test_best_threshold_picks_the_highest_f1():
    rows = sweep_threshold([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], [0.15, 0.5, 0.95])
    assert best_threshold(rows)["threshold"] == 0.5


def test_best_threshold_breaks_ties_toward_the_lower_value():
    """Two equally good thresholds -> prefer the more sensitive one."""
    rows = sweep_threshold([0.1, 0.9], [0, 1], [0.5, 0.6])
    # Both perfectly separate the two samples, so f1 ties at 1.0.
    assert rows[0]["f1"] == rows[1]["f1"] == 1.0
    assert best_threshold(rows)["threshold"] == 0.5


def test_min_recall_excludes_thresholds_that_miss_too_much():
    scores = [0.1, 0.4, 0.9]
    labels = [1, 1, 1]
    rows = sweep_threshold(scores, labels, [0.05, 0.5])
    # At 0.5 recall is only 1/3, so a floor of 0.9 rules it out.
    chosen = best_threshold(rows, min_recall=0.9)
    assert chosen["threshold"] == 0.05


def test_best_threshold_returns_none_when_nothing_qualifies():
    rows = sweep_threshold([0.1], [1], [0.5])
    assert best_threshold(rows, min_recall=0.99) is None


def test_best_threshold_can_optimise_precision_instead_of_f1():
    scores = [0.2, 0.6, 0.9]
    labels = [0, 1, 1]
    rows = sweep_threshold(scores, labels, [0.1, 0.8])
    # At 0.8 only the 0.9 sample fires and it's correct -> precision 1.0.
    assert best_threshold(rows, metric="precision")["threshold"] == 0.8
