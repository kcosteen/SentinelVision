"""Every decision boundary in the pipeline, in one place, with its provenance.

A threshold turns a continuous measurement into a yes/no, so each one is a claim
about where suspicious behaviour begins. Scattering them across modules made two
of them drift apart (phone confidence was 0.5 in `feature_extractor.py` and 0.6
in `object_tracker.py`) and made it impossible to say which had been measured and
which was a guess.

**CALIBRATED** means swept against labelled data; the command that produced it is
recorded so the number can be re-derived or challenged. **UNCALIBRATED** means
somebody picked it, including when that somebody was a paper.

Re-run calibration whenever the model or the camera changes -- the right
threshold is a property of both, not a constant of nature.
"""

import os

# --- Object detection -------------------------------------------------------

# Prefer the Phase 2 fine-tune; fall back to stock COCO weights when it's absent
# (a fresh clone, or before the Kaggle run). The fallback is deliberately loud at
# call sites rather than silent -- the two models are not interchangeable.
FINETUNED_WEIGHTS = os.path.join("models", "detection", "proctoring_yolov8n_best.pt")
BASELINE_WEIGHTS = "yolov8n.pt"


def detector_weights():
    """Path to the best available detector."""
    return FINETUNED_WEIGHTS if os.path.exists(FINETUNED_WEIGHTS) else BASELINE_WEIGHTS


def using_finetuned():
    return os.path.exists(FINETUNED_WEIGHTS)


# CALIBRATED for the fine-tuned model, on 700 held-out proctoring images:
#   python -m src.calibration.calibrate_phone_conf \
#       --weights models/detection/proctoring_yolov8n_best.pt --finetuned \
#       --export-root data/detection/external/roboflow_online_proctoring --split valid
# Best F1 0.923 at 0.35 (precision 0.909, recall 0.936). The curve is flat from
# 0.25-0.50, so this is a plateau rather than a knife edge.
PHONE_CONF = 0.35

# The same sweep on the PRE-TRAINED model peaked at f1 0.193 -- the baseline is
# so weak that no threshold rescues it. Kept only so the fallback path isn't
# silently using a value calibrated for a different model.
PHONE_CONF_BASELINE = 0.15

# Other objects have not been calibrated individually. They share the phone's
# value for now, which is a guess, not a measurement.
OBJECT_CONF = PHONE_CONF


def phone_conf():
    """Confidence floor appropriate to whichever detector is loaded."""
    return PHONE_CONF if using_finetuned() else PHONE_CONF_BASELINE


# --- Head pose --------------------------------------------------------------

# CALIBRATED against 463 Gourier head-pose images with ground-truth angles:
#   python -m src.calibration.validate_head_pose
# |yaw| >= 30 deg separates "turned away" at f1 0.869 (precision 0.835,
# recall 0.905) from true |pan| >= 45 deg.
#
# WARNING: only the MAGNITUDE is trustworthy. Signed yaw suffers a
# cv2.RQDecomp3x3 sign flip and cannot distinguish left from right, and the
# estimate saturates past |pan| ~60 deg. Use abs(yaw); do not branch on its sign.
HEAD_YAW_LOOKING_AWAY = 30.0

# --- Uncalibrated -----------------------------------------------------------

# UNCALIBRATED. The value from the original EAR paper, never checked against our
# landmarks or our camera. src/calibration/calibrate_ear.py is written but has
# never been run.
EAR_CLOSED = 0.20
BLINK_MIN_FRAMES = 2

# UNCALIBRATED. Hand-picked iris-position ratios. No public gaze dataset was
# found with usable ground truth, so these remain guesses -- the head-yaw
# threshold above is the better-evidenced "looking away" signal.
GAZE_LEFT = 0.35
GAZE_RIGHT = 0.65

# UNCALIBRATED. Suspicion-score bands in proctor_analyzer.
SCORE_SUSPICIOUS = 30
SCORE_HIGH_RISK = 70
