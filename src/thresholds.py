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


# OPERATING POINT, not a calibration. 0.35 is the F1 optimum on the public
# proctoring val set and stays the right default for measurement. Live in a real
# room it is too low: dark rectangular background objects (a wall shelf, a
# keyboard, a monitor edge) score as `cell phone` in the 0.34-0.53 band, so the
# app cries phone continuously at a desk it has never seen. A real phone held up
# to the camera scores 0.72-0.79, well clear of that.
#
# 0.60 sits in the gap. It costs recall on phones that are half out of frame --
# those only reach 0.09-0.16 and were never detectable at any usable threshold
# anyway -- and buys a demo that does not false-alarm on furniture.
#
# This is a judgement about ONE camera in ONE room, so it is deliberately not
# called calibrated. The honest fix is hard negatives from the deployment
# environment; see the limitations section in README.
PHONE_CONF_LIVE = 0.60


def phone_conf(live=False):
    """Confidence floor appropriate to whichever detector is loaded.

    `live=True` returns the stricter operating point used by the real-time app.
    Measurement paths keep the calibrated value so reported numbers stay
    comparable to everything already published.
    """
    if not using_finetuned():
        return PHONE_CONF_BASELINE
    return PHONE_CONF_LIVE if live else PHONE_CONF


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

# CALIBRATED against 1,999 labelled eyes-open/closed face images (ODC-BY):
#   python -m src.calibration.calibrate_ear --limit 2000
# The two classes barely overlap -- Cohen's d 4.11, "well separated":
#   closed  n=1199  mean 0.080   p90 0.164
#   open    n= 800  mean 0.360   p10 0.287
# So everything between ~0.16 and ~0.29 is no-man's-land, and the exact value
# matters far less than being inside it. The F1 argmax is 0.25 (p 0.987 /
# r 0.976 / f1 0.982) but precision starts dropping immediately above it -- 131
# false positives by 0.30. 0.23 gives up 0.003 F1 (0.979) to sit nearer the
# middle of the gap, so a modest EAR shift from a different camera doesn't cross
# it. Picking the argmax of a curve measured on someone else's capture setup is
# the exact mistake documented for the phone detector in README.
# The old value, 0.20 from the original EAR paper, scored f1 0.968.
EAR_CLOSED = 0.23

# UNCALIBRATED. How many consecutive closed frames count as one blink.
BLINK_MIN_FRAMES = 2

# UNCALIBRATED. Hand-picked iris-position ratios. No public gaze dataset was
# found with usable ground truth, so these remain guesses -- the head-yaw
# threshold above is the better-evidenced "looking away" signal.
GAZE_LEFT = 0.35
GAZE_RIGHT = 0.65

# UNCALIBRATED. Suspicion-score bands in proctor_analyzer.
SCORE_SUSPICIOUS = 30
SCORE_HIGH_RISK = 70

# UNCALIBRATED. Points bled off per second while nothing is flagged, so the score
# reflects RECENT behaviour rather than whether anything ever happened. At 2.0 a
# one-off "Phone detected" (+50) clears in 25s, while sustained phone use re-fires
# every `cooldown` (10s) for +50 and easily outruns the 20 points decay sheds in
# that time -- so real behaviour still reaches High Risk and stays.
SCORE_DECAY_PER_SEC = 2.0
