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
    """With no head pose available, the gaze fallback still decides."""
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(object_results=[], face_count=1, gaze="LEFT")
    assert events == ["Looking away"]


def test_center_gaze_is_not_flagged():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(object_results=[], face_count=1, gaze="CENTER")
    assert events == []


# --- the calibrated head-yaw rule (|yaw| >= 30 deg, f1 0.869) ---------------

def test_head_yaw_past_threshold_is_looking_away():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(
        object_results=[], face_count=1, gaze="CENTER", head_yaw=45.0
    )
    assert events == ["Looking away"]


def test_head_yaw_sign_is_ignored():
    """Signed yaw is unreliable (RQDecomp3x3 flip), so only |yaw| may be used."""
    analyzer = ProctorAnalyzer()
    for yaw in (40.0, -40.0):
        events = ProctorAnalyzer().analyze(
            object_results=[], face_count=1, gaze="CENTER", head_yaw=yaw
        )
        assert events == ["Looking away"], f"yaw={yaw}"


def test_head_yaw_below_threshold_is_not_flagged():
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(
        object_results=[], face_count=1, gaze="CENTER", head_yaw=12.0
    )
    assert events == []


def test_head_yaw_outranks_the_uncalibrated_gaze():
    """The measured signal decides; the guessed one must not override it.

    Facing forward (yaw 5 deg) with a gaze ratio that reads LEFT is exactly the
    case the hand-picked 0.35/0.65 cut-points get wrong, so it must not flag.
    """
    analyzer = ProctorAnalyzer()
    events = analyzer.analyze(
        object_results=[], face_count=1, gaze="LEFT", head_yaw=5.0
    )
    assert events == []


def test_gaze_is_used_only_when_head_pose_is_missing():
    analyzer = ProctorAnalyzer()
    assert analyzer.analyze(
        object_results=[], face_count=1, gaze="RIGHT", head_yaw=None
    ) == ["Looking away"]


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


# ---------------------------------------------------------------------------
# decay(): the score must reflect RECENT behaviour, not "did anything ever happen"
# ---------------------------------------------------------------------------

def test_score_decays_while_nothing_is_flagged():
    analyzer = ProctorAnalyzer(now=0.0)
    analyzer.add_score("Phone detected", now=0.0)
    assert analyzer.score == 50

    analyzer.decay(now=10.0)          # 10s clean at 2.0/s
    assert analyzer.score == 30


def test_a_single_false_positive_does_not_pin_the_session():
    """The bug this fixes: 2s of warm-up false positives used to mean
    'Suspicious' for the entire run, however long and however clean."""
    analyzer = ProctorAnalyzer(now=0.0)
    analyzer.add_score("Phone detected", now=0.0)
    assert analyzer.get_status() == "Suspicious"

    analyzer.decay(now=30.0)
    assert analyzer.score == 0
    assert analyzer.get_status() == "Normal"


def test_score_never_goes_negative():
    analyzer = ProctorAnalyzer(now=0.0)
    analyzer.add_score("Looking away", now=0.0)
    analyzer.decay(now=10_000.0)
    assert analyzer.score == 0


def test_sustained_behaviour_still_outruns_decay():
    """Decay must not defang a real offender: events re-fire every cooldown
    (10s, +50) which comfortably beats the 20 points decay sheds in that time."""
    analyzer = ProctorAnalyzer(now=0.0)
    for t in range(0, 31, 10):
        analyzer.decay(now=float(t))
        analyzer.add_score("Phone detected", now=float(t))
    assert analyzer.get_status() == "High Risk"


def test_decay_is_a_no_op_at_zero():
    analyzer = ProctorAnalyzer(now=0.0)
    assert analyzer.decay(now=100.0) == 0
