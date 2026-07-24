"""
Unit tests for the evaluation metrics.

Yes — we test the code that measures our models too. If the metric is wrong,
every conclusion drawn from it is wrong, so this is worth pinning down.
"""

import pytest

from evaluation.metrics import (
    confusion_counts,
    precision,
    recall,
    f1_score,
    evaluate_binary,
)


def test_confusion_counts():
    #            clip:  1  2  3  4  5  6  7  8
    y_true = [1, 0, 0, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 1, 1, 0, 0]
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    # correct positives: clips 1, 6            -> tp = 2
    # predicted 1 but truth 0: clip 5          -> fp = 1
    # truth 1 but predicted 0: clip 4          -> fn = 1
    # everything else truth 0 predicted 0      -> tn = 4
    assert (tp, fp, fn, tn) == (2, 1, 1, 4)


def test_precision_recall_f1_formulas():
    assert precision(tp=2, fp=1) == pytest.approx(2 / 3)
    assert recall(tp=2, fn=1) == pytest.approx(2 / 3)
    assert f1_score(2 / 3, 2 / 3) == pytest.approx(2 / 3)


def test_metrics_are_zero_not_crash_on_empty():
    # No positive predictions -> precision is defined as 0.0 (not a crash).
    assert precision(tp=0, fp=0) == 0.0
    assert recall(tp=0, fn=0) == 0.0
    assert f1_score(0.0, 0.0) == 0.0


def test_evaluate_binary_perfect_prediction():
    result = evaluate_binary([1, 0, 1, 0], [1, 0, 1, 0])
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["support"] == 2
