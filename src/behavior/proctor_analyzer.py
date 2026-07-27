"""Turns per-frame signals into events and a running suspicion score.

Every decision boundary used here lives in src/thresholds.py, which records for
each one whether it was measured against data or picked by hand.

The "Looking away" rule is the one worth explaining in an interview. Two signals
could drive it:

* **Head yaw** -- CALIBRATED. `src/calibration/validate_head_pose.py` ran the
  real PnP solver over 463 Gourier head-pose images whose filenames encode the
  true pan angle, and swept the cut-point: |yaw| >= 30 deg separates "turned
  away" at f1 0.869 (precision 0.835, recall 0.905).
* **Iris gaze ratio** -- UNCALIBRATED. 0.35/0.65 were picked by hand; no public
  gaze dataset with usable ground truth turned up to check them against.

So head yaw is preferred whenever it's available, and gaze is only a fallback
for frames where head pose could not be solved. Only the MAGNITUDE of the yaw is
used: signed yaw suffers a cv2.RQDecomp3x3 sign flip and cannot tell left from
right, which is fine for "is the head turned away?" and useless for "which way?".
"""

import time

from src.thresholds import (
    HEAD_YAW_LOOKING_AWAY,
    SCORE_HIGH_RISK,
    SCORE_SUSPICIOUS,
)


class ProctorAnalyzer:

    def __init__(self):
        self.score = 0

        self.event_history = {}

        self.cooldown = 10


    def add_score(self, event):

        scores = {
            "No face detected": 20,
            "Multiple people detected": 40,
            "Looking away": 10,
            "Phone detected": 50
        }

        current_time = time.time()


        if event in self.event_history:

            last_time = self.event_history[event]

            if current_time - last_time < self.cooldown:
                return 0


        self.event_history[event] = current_time


        points = scores.get(event, 0)

        self.score += points


        return points



    def analyze(
        self,
        object_results,
        face_count,
        gaze,
        head_yaw=None
    ):
        """Raw per-frame signals -> a list of event names.

        `head_yaw` is degrees from the PnP solver, or None when no face pose was
        solved for this frame. See the module docstring for why it outranks gaze.
        """

        events = []


        # Face checks
        if face_count == 0:
            events.append("No face detected")


        if face_count > 1:
            events.append("Multiple people detected")


        # Looking away: use the measured signal when we have it, the guessed
        # one only when we don't. abs() because the yaw SIGN is unreliable.
        if head_yaw is not None:
            if abs(head_yaw) >= HEAD_YAW_LOOKING_AWAY:
                events.append("Looking away")

        elif gaze is not None and gaze != "CENTER":
            events.append("Looking away")


        # Object checks
        for obj in object_results:

            if obj["label"] == "cell phone":
                events.append("Phone detected")


        return events
    
    def get_status(self):
        """Score -> label. Bands are UNCALIBRATED (see src/thresholds.py)."""

        if self.score < SCORE_SUSPICIOUS:
            return "Normal"

        elif self.score < SCORE_HIGH_RISK:
            return "Suspicious"

        else:
            return "High Risk"