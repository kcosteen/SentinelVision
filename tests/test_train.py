"""
Unit tests for the Phase 1 training harness.

These check the wiring, not the science: the pipeline fits + predicts, and
`evaluate` handles behaviours with too little data instead of crashing.
"""

import numpy as np
import pandas as pd

from src.data.build_dataset import FEATURE_FIELDS
from src.data.labels import LABEL_COLUMNS
from src.models.train import evaluate, make_pipeline


def _dataset(n_clips=6, windows_per_clip=3, positive_clips=("c0", "c1", "c2")):
    """A small synthetic windowed dataset with a learnable phone signal."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_clips):
        clip_id = f"c{c}"
        is_phone = clip_id in positive_clips
        for _ in range(windows_per_clip):
            row = {field: float(rng.random()) for field in FEATURE_FIELDS}
            # Make phone_frac separable so the forest has something to learn.
            row["phone_frac"] = rng.uniform(0.3, 0.6) if is_phone else 0.0
            row["clip_id"] = clip_id
            for col in LABEL_COLUMNS:
                row[col] = 0
            row["label_phone"] = int(is_phone)
            rows.append(row)
    return pd.DataFrame(rows)


def test_pipeline_fits_and_predicts():
    df = _dataset()
    pipe = make_pipeline().fit(df[FEATURE_FIELDS], df["label_phone"])
    preds = pipe.predict(df[FEATURE_FIELDS])
    assert len(preds) == len(df)
    assert set(np.unique(preds)) <= {0, 1}


def test_pipeline_handles_missing_features():
    # A NaN (no-face frame that survived aggregation) must not crash fitting.
    df = _dataset()
    df.loc[0, "gaze_ratio_mean"] = np.nan
    pipe = make_pipeline().fit(df[FEATURE_FIELDS], df["label_phone"])
    assert pipe.predict(df[FEATURE_FIELDS].iloc[[0]]).shape == (1,)


def test_evaluate_scores_a_learnable_behaviour():
    results = evaluate(_dataset(), n_splits=3)
    assert results["phone"]["status"] == "ok"
    assert 0.0 <= results["phone"]["f1"] <= 1.0


def test_evaluate_skips_behaviours_without_data():
    results = evaluate(_dataset(), n_splits=3)
    # No absent / multiple_people examples in the synthetic set.
    assert results["absent"]["status"] == "insufficient_data"
    assert results["absent"]["pos_windows"] == 0
