"""
Unit tests for the demo inference helpers.

The video/model path is validated by running the app; here we pin the pure
result-shaping logic that turns per-window predictions into a demo summary:
interval merging, the rollup, and the headline verdict thresholds.
"""

import pandas as pd

from src.inference.analyze import _flagged_intervals, overall_verdict, summarize


def _windows(flags, probs=None, behaviour="looking_away"):
    """Build a minimal windows DataFrame with 2s windows at 1s steps."""
    probs = probs if probs is not None else [1.0 if f else 0.0 for f in flags]
    return pd.DataFrame({
        "window_start_sec": [float(i) for i in range(len(flags))],
        "window_end_sec": [float(i + 2) for i in range(len(flags))],
        f"{behaviour}_flag": flags,
        f"{behaviour}_prob": probs,
    })


def test_intervals_merge_consecutive_flags():
    # Windows 0-2, 1-3, 2-4 flagged -> one merged span 0..4.
    windows = _windows([1, 1, 1, 0])
    assert _flagged_intervals(windows, "looking_away") == [(0.0, 4.0)]


def test_intervals_stay_separate_when_gapped():
    # Flag, gap, flag -> two distinct spans.
    windows = _windows([1, 0, 0, 1])
    assert _flagged_intervals(windows, "looking_away") == [(0.0, 2.0), (3.0, 5.0)]


def test_summarize_counts_and_peak():
    windows = _windows([1, 1, 0, 0], probs=[0.9, 0.8, 0.1, 0.2])
    summary = summarize(windows, ["looking_away"])["looking_away"]
    assert summary["flagged_windows"] == 2
    assert summary["total_windows"] == 4
    assert summary["flagged_fraction"] == 0.5
    assert summary["max_prob"] == 0.9


def test_verdict_thresholds():
    assert overall_verdict({"x": {"flagged_fraction": 0.0}})[0] == "Appears normal"
    assert overall_verdict({"x": {"flagged_fraction": 0.2}})[0] == "Suspicious"
    assert overall_verdict({"x": {"flagged_fraction": 0.8}})[0] == "High risk"


def test_verdict_uses_worst_behaviour():
    summary = {"looking_away": {"flagged_fraction": 0.05},
               "phone": {"flagged_fraction": 0.9}}
    assert overall_verdict(summary)[0] == "High risk"
