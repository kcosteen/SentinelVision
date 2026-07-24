"""
Unit tests for the behavior-analysis engine (the current rules-based "brain").

These tests pin down the *contract* of the analyzer: which events it raises,
how it scores them, how the per-event cooldown works, and how the score maps to
a risk status. Having them means that when Phase 1 replaces these rules with a
trained model, we can prove the new version preserves (or intentionally changes)
this behavior.
"""

import time

from src.behavior.proctor_analyzer import ProctorAnalyzer


# ---------------------------------------------------------------------------
# analyze(): raw signals -> list of events
# ---------------------------------------------------------------------------

def test_no_face_raises_event():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(object_results=[], face_count=0, gaze="CENTER")
    assert events == ["No face detected"]


def test_multiple_faces_raises_event():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(object_results=[], face_count=3, gaze="CENTER")
    assert events == ["Multiple people detected"]


def test_looking_away_raises_event():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(object_results=[], face_count=1, gaze="LEFT")
    assert events == ["Looking away"]


def test_center_gaze_is_not_flagged():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(object_results=[], face_count=1, gaze="CENTER")
    assert events == []


def test_phone_object_raises_event():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(
        object_results=[{"label": "cell phone", "confidence": 0.9}],
        face_count=1,
        gaze="CENTER",
    )
    assert events == ["Phone detected"]


# ---------------------------------------------------------------------------
# add_score(): scoring + cooldown
# ---------------------------------------------------------------------------

def test_add_score_returns_expected_points():
    analyzer = ProctorAnalyzer()
    assert analyzer.add_score("Phone detected") == 50
    assert analyzer.score == 50


def test_unknown_event_scores_zero():
    analyzer = ProctorAnalyzer()
    assert analyzer.add_score("Some unknown event") == 0
    assert analyzer.score == 0


def test_cooldown_suppresses_repeat_event():
    analyzer = ProctorAnalyzer()
    first = analyzer.add_score("Looking away")
    second = analyzer.add_score("Looking away")  # immediately again
    assert first == 10
    assert second == 0            # suppressed by the 10s cooldown
    assert analyzer.score == 10   # score only counted once


def test_score_counts_again_after_cooldown_expires():
    analyzer = ProctorAnalyzer()
    analyzer.add_score("Looking away")
    # Simulate that the last occurrence was 11 seconds ago (past the cooldown)
    # instead of sleeping in the test.
    analyzer.event_history["Looking away"] = time.time() - 11
    assert analyzer.add_score("Looking away") == 10
    assert analyzer.score == 20


# ---------------------------------------------------------------------------
# get_status(): score -> risk band
# ---------------------------------------------------------------------------

def test_status_bands():
    analyzer = ProctorAnalyzer()

    analyzer.score = 0
    assert analyzer.get_status() == "Normal"

    analyzer.score = 29
    assert analyzer.get_status() == "Normal"

    analyzer.score = 30
    assert analyzer.get_status() == "Suspicious"

    analyzer.score = 69
    assert analyzer.get_status() == "Suspicious"

    analyzer.score = 70
    assert analyzer.get_status() == "High Risk"
