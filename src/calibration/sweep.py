"""Sweep a decision threshold and report what it costs you.

Every threshold in this project turns a continuous score into a yes/no: EAR into
"eyes closed", detector confidence into "phone present". Picking one by eye means
picking a precision/recall trade-off blind. Sweeping makes the trade-off visible:
run every candidate threshold against labelled data, and read off where you want
to sit.

The metric maths is reused from `evaluation/metrics.py` rather than re-derived --
one definition of precision in the project, not two.

**There is no single "correct" threshold**, which is the point of reporting the
whole curve rather than one number. F1 is the default because it balances the two
error types, but proctoring is not symmetric: a false accusation of cheating is
far worse than a missed phone, so you may deliberately choose a higher-precision
point than F1 suggests. `best_threshold(..., metric="precision", min_recall=0.5)`
supports that -- "the most precise threshold that still catches half of them".
"""

from typing import Dict, List, Optional, Sequence

from evaluation.metrics import confusion_counts, f1_score, precision, recall


def candidate_thresholds(start=0.05, stop=0.95, step=0.05):
    """Inclusive range of thresholds, avoiding float drift in the endpoints."""
    n_steps = int(round((stop - start) / step))
    return [round(start + i * step, 4) for i in range(n_steps + 1)]


def sweep_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    thresholds: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """Score a binary decision at each threshold.

    `scores[i]` is the continuous measurement, `labels[i]` the 0/1 truth.
    A sample counts as predicted-positive when `score >= threshold`.

    Returns one row per threshold with tp/fp/fn/tn and precision/recall/f1.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels differ in length: {len(scores)} vs {len(labels)}"
        )

    rows = []
    for threshold in (thresholds if thresholds is not None else candidate_thresholds()):
        predictions = [1 if score >= threshold else 0 for score in scores]
        tp, fp, fn, tn = confusion_counts(labels, predictions)
        prec = precision(tp, fp)
        rec = recall(tp, fn)
        rows.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec,
            "recall": rec,
            "f1": f1_score(prec, rec),
        })
    return rows


def best_threshold(
    rows: Sequence[Dict[str, float]],
    metric: str = "f1",
    min_recall: float = 0.0,
) -> Optional[Dict[str, float]]:
    """The best row by `metric`, among those meeting `min_recall`.

    Ties break toward the LOWER threshold. For a detector that means preferring
    the more sensitive setting when two are otherwise equal -- in proctoring,
    a flagged frame gets reviewed, a missed one is gone.
    """
    eligible = [row for row in rows if row["recall"] >= min_recall]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (-row[metric], row["threshold"]))


def format_sweep(rows: Sequence[Dict[str, float]], highlight: Optional[float] = None) -> str:
    """Render the sweep as a table, marking the chosen threshold with '<--'."""
    header = (f"{'thresh':>7}{'TP':>7}{'FP':>6}{'FN':>6}"
              f"{'precision':>11}{'recall':>9}{'f1':>7}")
    lines = [header, "-" * len(header)]
    for row in rows:
        marker = "  <--" if highlight is not None and row["threshold"] == highlight else ""
        lines.append(
            f"{row['threshold']:>7.2f}{int(row['tp']):>7}{int(row['fp']):>6}"
            f"{int(row['fn']):>6}{row['precision']:>11.3f}{row['recall']:>9.3f}"
            f"{row['f1']:>7.3f}{marker}"
        )
    return "\n".join(lines)


def summarise_choice(rows, chosen, current, name, metric="f1"):
    """Print the recommendation next to whatever the code uses today."""
    print(format_sweep(rows, highlight=chosen["threshold"] if chosen else None))

    if chosen is None:
        print("\nNo threshold met the constraint -- loosen --min-recall.")
        return

    # An argmax sitting on the edge of the swept range is a boundary artifact:
    # the real optimum may lie outside it, so the "best" reported here is only
    # the best of what was looked at. Say so rather than quietly implying it.
    edges = (rows[0]["threshold"], rows[-1]["threshold"])
    if chosen["threshold"] in edges and len(rows) > 1:
        print(f"\nWARNING: the best value ({chosen['threshold']:.2f}) is at the edge "
              f"of the swept range\n[{edges[0]:.2f}, {edges[1]:.2f}] -- the true "
              f"optimum may lie beyond it. Re-run with a wider\n--start / --stop "
              f"before trusting this number.")

    print(f"\nBest by {metric}: {name} = {chosen['threshold']:.2f}  "
          f"(precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f}, "
          f"f1 {chosen['f1']:.3f})")

    # The comparison that makes this actionable: is the hand-picked value wrong?
    current_row = min(rows, key=lambda row: abs(row["threshold"] - current))
    print(f"Currently in code: {name} = {current:.2f}  "
          f"(precision {current_row['precision']:.3f}, "
          f"recall {current_row['recall']:.3f}, f1 {current_row['f1']:.3f})")

    delta = chosen["f1"] - current_row["f1"]
    if abs(delta) < 0.01:
        print("=> The hand-picked value is already within 0.01 F1 of optimal. Leave it.")
    else:
        print(f"=> Moving to {chosen['threshold']:.2f} gains {delta:+.3f} F1.")
