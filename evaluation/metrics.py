"""
Classification metrics, implemented from scratch.

We could import these from scikit-learn, but writing them out (a) keeps the
project dependency-light and (b) forces us to actually understand what the
numbers mean — which is exactly what an interviewer will ask about.

Every metric is derived from the four cells of a binary confusion matrix:

                   predicted 1        predicted 0
    actual 1     TP (true pos.)     FN (false neg.)
    actual 0     FP (false pos.)    TN (true neg.)

    precision = TP / (TP + FP)   "of the alerts I raised, how many were real?"
    recall    = TP / (TP + FN)   "of the real events, how many did I catch?"
    f1        = harmonic mean of precision and recall
    accuracy  = (TP + TN) / total
"""

from typing import Dict, Sequence, Tuple


def confusion_counts(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive: int = 1,
) -> Tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for a binary problem."""
    tp = fp = fn = tn = 0
    for actual, predicted in zip(y_true, y_pred):
        if predicted == positive and actual == positive:
            tp += 1
        elif predicted == positive and actual != positive:
            fp += 1
        elif predicted != positive and actual == positive:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision(tp: int, fp: int) -> float:
    # Guard against 0/0: if we made no positive predictions, precision is 0.
    return tp / (tp + fp) if (tp + fp) else 0.0


def recall(tp: int, fn: int) -> float:
    # Guard against 0/0: if there were no positives to find, recall is 0.
    return tp / (tp + fn) if (tp + fn) else 0.0


def f1_score(prec: float, rec: float) -> float:
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    total = tp + fp + fn + tn
    return (tp + tn) / total if total else 0.0


def evaluate_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Compute the full metric set for one binary behavior."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1_score(prec, rec),
        "accuracy": accuracy(tp, fp, fn, tn),
        # "support" = how many real positives exist in the ground truth.
        "support": sum(1 for t in y_true if t == 1),
    }


def format_report(results_by_label: Dict[str, Dict[str, float]]) -> str:
    """Render a metrics table like scikit-learn's classification_report."""
    header = f"{'behavior':<18}{'precision':>10}{'recall':>9}{'f1':>7}{'support':>9}"
    lines = [header, "-" * len(header)]
    for label, m in results_by_label.items():
        lines.append(
            f"{label:<18}{m['precision']:>10.2f}{m['recall']:>9.2f}"
            f"{m['f1']:>7.2f}{int(m['support']):>9}"
        )
    return "\n".join(lines)
