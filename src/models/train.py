"""
Train + evaluate the Phase 1 behaviour model -- the flagship ML step.

This replaces the hand-tuned rules in `proctor_analyzer.py` with a model that
*learns* how to weigh the windowed features (gaze/head variance, blink rate,
phone fraction, ...) into a per-behaviour decision.

    python -m src.models.train                 # train on data/dataset.csv
    python -m src.models.train --folds 5        # grouped CV folds

Two methodology choices that make the reported numbers honest -- and that are the
whole point of a portfolio project:

* **Split by clip, not by window.** Windows from the same clip are near-duplicates.
  We use GroupKFold on `clip_id`, so every test fold contains *clips the model
  never trained on*. Splitting rows at random would leak the answer and inflate F1.
  The real sample size for generalisation is the number of *clips*, not windows.
* **Impute inside the pipeline.** Median imputation for missing (no-face) features
  is fit on the training fold only, so test-fold statistics never leak in.

We report **per-behaviour precision / recall / F1**, not accuracy: cheating is
rare, so a model that always predicts "normal" scores high accuracy while catching
nothing. Precision/recall expose that; accuracy hides it.

Behaviours with no positive examples yet (e.g. absent / multiple_people until you
record clips for them) are skipped automatically -- add data and they light up
with no code change.
"""

import argparse
import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

import pandas as pd

from src.data.build_dataset import FEATURE_FIELDS
from src.data.labels import BEHAVIOR_LABELS

# A behaviour needs positives in at least this many distinct clips before a
# grouped CV score means anything -- one clip can't tell you it generalises.
MIN_POSITIVE_CLIPS = 2


def make_pipeline(random_state=0):
    """Median-impute missing features, then a small, interpretable random forest.

    The imputer sits INSIDE the pipeline so cross-validation fits it per training
    fold -- no leakage. A forest needs no feature scaling and exposes feature
    importances, which is exactly the interpretability a first model should have.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",   # counteract the normal-heavy class imbalance
            random_state=random_state,
        )),
    ])


def evaluate_label(X, y, groups, n_splits, random_state=0):
    """Grouped out-of-fold predictions -> (precision, recall, f1)."""
    preds = cross_val_predict(
        make_pipeline(random_state), X, y, groups=groups,
        cv=GroupKFold(n_splits=n_splits),
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0,
    )
    return precision, recall, f1


def evaluate(df, n_splits):
    """Per-behaviour grouped-CV metrics; skips behaviours without enough data."""
    X = df[FEATURE_FIELDS]
    groups = df["clip_id"]
    n_clips = groups.nunique()

    results = {}
    for behaviour in BEHAVIOR_LABELS:
        y = df[f"label_{behaviour}"]
        pos_windows = int(y.sum())
        pos_clips = int(df.loc[y == 1, "clip_id"].nunique())

        if pos_clips < MIN_POSITIVE_CLIPS:
            results[behaviour] = {
                "status": "insufficient_data",
                "pos_windows": pos_windows,
                "pos_clips": pos_clips,
            }
            continue

        folds = min(n_splits, n_clips)
        precision, recall, f1 = evaluate_label(X, y, groups, folds)
        results[behaviour] = {
            "status": "ok",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pos_windows": pos_windows,
            "pos_clips": pos_clips,
            "folds": folds,
        }
    return results


def top_features(pipeline, k=5):
    """The k features the trained forest leaned on most (name, importance)."""
    importances = pipeline.named_steps["forest"].feature_importances_
    ranked = sorted(zip(FEATURE_FIELDS, importances), key=lambda p: p[1], reverse=True)
    return ranked[:k]


def print_report(results):
    print("\nPhase 1 behaviour model -- grouped CV (split by clip)\n")
    print(f"{'behaviour':<18}{'P':>7}{'R':>7}{'F1':>7}   pos(win/clip)")
    print("-" * 56)
    for behaviour, r in results.items():
        if r["status"] == "ok":
            print(f"{behaviour:<18}{r['precision']:>7.2f}{r['recall']:>7.2f}"
                  f"{r['f1']:>7.2f}   {r['pos_windows']:>4} / {r['pos_clips']}")
        else:
            note = f"only {r['pos_clips']} clip(s)" if r["pos_windows"] else "no clips yet"
            print(f"{behaviour:<18}{'--':>7}{'--':>7}{'--':>7}   {note}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=os.path.join("data", "dataset.csv"))
    parser.add_argument("--folds", type=int, default=5, help="grouped CV folds")
    parser.add_argument("--out-model", default=os.path.join("models", "behavior_model.joblib"))
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Loaded {len(df)} windows from {df['clip_id'].nunique()} clips.")

    results = evaluate(df, args.folds)
    print_report(results)

    # Fit a final model per trained behaviour on ALL data, and report what drove it.
    X = df[FEATURE_FIELDS]
    models = {}
    print("\nTop features per behaviour (final model on all data):")
    for behaviour, r in results.items():
        if r["status"] != "ok":
            continue
        pipeline = make_pipeline().fit(X, df[f"label_{behaviour}"])
        models[behaviour] = pipeline
        drivers = ", ".join(f"{name} {imp:.2f}" for name, imp in top_features(pipeline))
        print(f"  {behaviour:<18} {drivers}")

    if models:
        os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
        joblib.dump(models, args.out_model)
        print(f"\nSaved {len(models)} trained behaviour model(s) -> {args.out_model}")
    else:
        print("\nNo behaviour had enough data to train yet -- record more clips.")


if __name__ == "__main__":
    main()
