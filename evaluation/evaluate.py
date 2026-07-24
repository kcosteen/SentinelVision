"""
Evaluate the proctor's behavior predictions against human-labeled ground truth.

Idea
----
Each *clip* (a short recording) is labeled by a human for a set of behaviors,
e.g. was the person `looking_away`, was a `phone_present`, were there
`multiple_people`. Separately, we run the pipeline over the same clips and record
what it *predicted*. This script joins the two on `clip_id` and reports, per
behavior, how well the predictions matched reality (precision / recall / F1).

Both files are CSVs with the same columns, one row per clip:

    clip_id,looking_away,phone_present,multiple_people
    clip_001,1,0,0
    ...

Usage
-----
    python -m evaluation.evaluate
    python -m evaluation.evaluate --truth path/to/truth.csv --pred path/to/pred.csv
"""

import argparse
import csv
import os
from typing import Dict, List

from evaluation.metrics import evaluate_binary, format_report

HERE = os.path.dirname(__file__)
DEFAULT_TRUTH = os.path.join(HERE, "sample_ground_truth.csv")
DEFAULT_PRED = os.path.join(HERE, "sample_predictions.csv")


def load_labels(path: str) -> Dict[str, Dict[str, int]]:
    """Read a labels CSV into {clip_id: {behavior: 0/1, ...}}."""
    rows: Dict[str, Dict[str, int]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = row.pop("clip_id")
            rows[clip_id] = {behavior: int(value) for behavior, value in row.items()}
    return rows


def build_aligned_columns(
    truth: Dict[str, Dict[str, int]],
    pred: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, List[int]]]:
    """For every behavior, collect aligned (y_true, y_pred) lists.

    Only clips present in *both* files are scored, so a missing prediction
    can't silently be treated as a correct "0".
    """
    shared_clips = sorted(set(truth) & set(pred))
    if not shared_clips:
        raise ValueError("No clip_ids are shared between the two files.")

    behaviors = list(next(iter(truth.values())).keys())
    columns: Dict[str, Dict[str, List[int]]] = {
        b: {"y_true": [], "y_pred": []} for b in behaviors
    }
    for clip_id in shared_clips:
        for behavior in behaviors:
            columns[behavior]["y_true"].append(truth[clip_id][behavior])
            columns[behavior]["y_pred"].append(pred[clip_id][behavior])
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", default=DEFAULT_TRUTH, help="ground-truth CSV")
    parser.add_argument("--pred", default=DEFAULT_PRED, help="predictions CSV")
    args = parser.parse_args()

    truth = load_labels(args.truth)
    pred = load_labels(args.pred)
    columns = build_aligned_columns(truth, pred)

    results = {
        behavior: evaluate_binary(cols["y_true"], cols["y_pred"])
        for behavior, cols in columns.items()
    }

    n_clips = len(set(truth) & set(pred))
    print(f"\nEvaluated {n_clips} clips\n")
    print(format_report(results))
    print()


if __name__ == "__main__":
    main()
