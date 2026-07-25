"""
Unit tests for the detection metrics (IoU, matching, AP@0.5).

Same reasoning as tests/test_metrics.py: Phase 2's headline claim is a *number*
comparing two detectors, so the code producing that number has to be pinned down.
These cases are all hand-computable -- if a test fails you can work out the right
answer on paper, which is the point.
"""

import pytest

from src.detection.detection_metrics import (
    iou,
    match_image,
    average_precision,
    evaluate_detections,
)


def test_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_touching_edges_is_zero():
    # Sharing a border is not overlap: intersection area is 0.
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes offset by 5 in x: intersection 5*10=50, union 200-50=150.
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_match_image_counts_a_good_prediction_as_true_positive():
    truths = [(0, 0, 10, 10)]
    predictions = [((0, 0, 10, 10), 0.9)]
    assert match_image(predictions, truths) == [(0.9, 1)]


def test_match_image_rejects_prediction_below_iou_threshold():
    # IoU here is 50/150 = 0.33, under the 0.5 bar -> false positive.
    truths = [(0, 0, 10, 10)]
    predictions = [((5, 0, 15, 10), 0.9)]
    assert match_image(predictions, truths) == [(0.9, 0)]


def test_each_truth_can_only_be_claimed_once():
    """Two boxes on one phone must not score two true positives."""
    truths = [(0, 0, 10, 10)]
    predictions = [((0, 0, 10, 10), 0.9), ((0, 0, 10, 10), 0.8)]
    matches = match_image(predictions, truths)
    # Higher confidence claims it; the duplicate becomes a false positive.
    assert matches == [(0.9, 1), (0.8, 0)]


def test_missed_truth_produces_no_match_row():
    """A ground-truth box nobody predicted contributes no row -- it hurts recall."""
    truths = [(0, 0, 10, 10), (100, 100, 110, 110)]
    predictions = [((0, 0, 10, 10), 0.9)]
    matches = match_image(predictions, truths)
    assert matches == [(0.9, 1)]
    # Recall is 1/2 even though every prediction was correct.
    ap, _, rec = average_precision(matches, n_truths=len(truths))
    assert rec == pytest.approx(0.5)


def test_average_precision_perfect_detector_is_one():
    matches = [(0.9, 1), (0.8, 1)]
    ap, prec, rec = average_precision(matches, n_truths=2)
    assert ap == pytest.approx(1.0)
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)


def test_average_precision_with_no_ground_truth_is_zero_not_crash():
    assert average_precision([(0.9, 0)], n_truths=0) == (0.0, 0.0, 0.0)


def test_average_precision_penalises_a_confident_false_positive():
    """A wrong box ranked above a right one drags AP below 1.0."""
    matches = [(0.95, 0), (0.90, 1)]
    ap, _, rec = average_precision(matches, n_truths=1)
    assert rec == pytest.approx(1.0)   # the real box was still found
    assert ap == pytest.approx(0.5)    # but precision at that recall is only 1/2


def test_confidence_ordering_drives_the_curve():
    """Same boxes, better-ordered confidences -> strictly better AP."""
    good = average_precision([(0.9, 1), (0.5, 0)], n_truths=1)[0]
    bad = average_precision([(0.9, 0), (0.5, 1)], n_truths=1)[0]
    assert good > bad


def test_evaluate_detections_aggregates_across_images():
    per_image = [
        ([((0, 0, 10, 10), 0.9)], [(0, 0, 10, 10)]),          # hit
        ([((0, 0, 10, 10), 0.8)], [(50, 50, 60, 60)]),        # miss (no overlap)
        ([], [(0, 0, 10, 10)]),                               # nothing predicted
    ]
    result = evaluate_detections(per_image)
    assert result["n_images"] == 3
    assert result["n_truths"] == 3
    assert result["n_predictions"] == 2
    # Only 1 of 3 real boxes was found -> recall is exactly 1/3.
    assert result["recall"] == pytest.approx(1 / 3)
