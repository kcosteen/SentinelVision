"""Object-detection metrics (IoU, AP@0.5), implemented from scratch.

Ultralytics reports mAP for models it trained, but Phase 2's headline claim is a
*comparison*: fine-tuned vs. the pre-trained baseline, on the same held-out
images. The baseline predicts COCO class 67 ('cell phone') while our model
predicts class 0 ('phone'), so no single `model.val()` call can score both. Doing
the matching ourselves sidesteps the class-id mismatch and -- as in
`evaluation/metrics.py` -- keeps the number something we can actually explain.

Classification asks "was this label right?". Detection also asks "was it in the
right *place*?", so a prediction only counts as a true positive when it overlaps
a ground-truth box by at least an IoU threshold (0.5 by convention) AND that box
hasn't already been claimed by a higher-confidence prediction.

    IoU = area(intersection) / area(union)

Average Precision then summarises the whole precision/recall trade-off: sweep the
confidence threshold from high to low, and average precision over recall levels.
That makes AP threshold-independent, which is what lets us compare two models
that are confident in different ways.
"""

from typing import Dict, List, Sequence, Tuple

# A box is (x1, y1, x2, y2) in pixels. A prediction adds a confidence.
Box = Tuple[float, float, float, float]
Prediction = Tuple[Box, float]


def iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # The overlap rectangle; if the boxes miss each other, width or height is <= 0.
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_image(
    predictions: Sequence[Prediction],
    truths: Sequence[Box],
    iou_threshold: float = 0.5,
) -> List[Tuple[float, int]]:
    """Greedily match one image's predictions to its ground-truth boxes.

    Returns [(confidence, is_true_positive)] in descending confidence order.

    Greedy-by-confidence is the standard rule (COCO/Pascal VOC): the most
    confident prediction gets first claim on a box, and each ground-truth box can
    be claimed once. Without the "claim once" rule, a model could spam ten boxes
    on one phone and score ten true positives.
    """
    ordered = sorted(predictions, key=lambda p: p[1], reverse=True)
    claimed = [False] * len(truths)
    results = []

    for box, confidence in ordered:
        best_iou = 0.0
        best_index = -1
        for index, truth in enumerate(truths):
            if claimed[index]:
                continue
            overlap = iou(box, truth)
            if overlap > best_iou:
                best_iou, best_index = overlap, index

        if best_index >= 0 and best_iou >= iou_threshold:
            claimed[best_index] = True
            results.append((confidence, 1))
        else:
            # Either it overlapped nothing, or only boxes already claimed.
            results.append((confidence, 0))

    return results


def average_precision(
    matches: Sequence[Tuple[float, int]],
    n_truths: int,
) -> Tuple[float, float, float]:
    """AP@0.5 plus the precision/recall at the operating point of best F1.

    `matches` is every prediction across ALL images as (confidence, is_tp);
    `n_truths` is the total number of ground-truth boxes.

    AP uses all-point interpolation (Pascal VOC 2010+ / COCO): walk the
    predictions from most to least confident, recording the precision/recall
    curve, then integrate it after making precision monotonically decreasing.
    The monotonic fix stops a lucky late true positive from creating a spurious
    bump in the curve.
    """
    if n_truths == 0:
        return 0.0, 0.0, 0.0

    ordered = sorted(matches, key=lambda m: m[0], reverse=True)

    precisions, recalls = [], []
    best_f1 = best_precision = best_recall = 0.0
    tp = fp = 0

    for _, is_tp in ordered:
        tp += is_tp
        fp += 1 - is_tp
        precision = tp / (tp + fp)
        recall = tp / n_truths
        precisions.append(precision)
        recalls.append(recall)

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1, best_precision, best_recall = f1, precision, recall

    if not precisions:
        return 0.0, 0.0, 0.0

    # Make precision monotonically decreasing from the right.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Integrate: sum precision * (change in recall) over the curve.
    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        ap += precision * (recall - previous_recall)
        previous_recall = recall

    return ap, best_precision, best_recall


def evaluate_detections(
    per_image: Sequence[Tuple[Sequence[Prediction], Sequence[Box]]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Score a whole dataset: [(predictions, truths)] per image -> metric dict."""
    all_matches: List[Tuple[float, int]] = []
    n_truths = 0

    for predictions, truths in per_image:
        n_truths += len(truths)
        all_matches.extend(match_image(predictions, truths, iou_threshold))

    ap, precision, recall = average_precision(all_matches, n_truths)
    return {
        "ap50": ap,
        "precision": precision,
        "recall": recall,
        "n_truths": n_truths,
        "n_predictions": len(all_matches),
        "n_images": len(per_image),
    }
