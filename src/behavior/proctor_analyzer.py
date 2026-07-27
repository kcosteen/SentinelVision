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
    SCORE_DECAY_PER_SEC,
    STARTUP_GRACE_SECONDS,
    SCORE_HIGH_RISK,
    SCORE_SUSPICIOUS,
)


# Weights track how much each signal is TRUSTED, not just how bad the behaviour
# is. "Gaze off screen" is the cheapest because it rests on the one threshold
# still uncalibrated -- see the note in analyze(). Module-level so the HUD can
# show each alert's cost without keeping a second copy that could drift.
EVENT_POINTS = {
    "No face detected": 20,
    "Multiple people detected": 40,
    "Looking away": 10,
    "Gaze off screen": 5,
    "Phone detected": 50,
}


class ProctorAnalyzer:

    def __init__(self, now=None):
        self.score = 0

        self.event_history = {}

        self.cooldown = 10

        # Wall-clock of the last decay step. Injectable so tests need no sleeps.
        self._last_decay = now if now is not None else time.time()

        self.started_at = self._last_decay


    def add_score(self, event, now=None):

        # Weights track how much each signal is TRUSTED, not just how bad the
        # behaviour is. "Gaze off screen" is the cheapest because it rests on the
        # one threshold still uncalibrated -- see the note in analyze().
        scores = {
            "No face detected": 20,
            "Multiple people detected": 40,
            "Looking away": 10,
            "Gaze off screen": 5,
            "Phone detected": 50
        }

        current_time = now if now is not None else time.time()


        if event in self.event_history:

            last_time = self.event_history[event]

            if current_time - last_time < self.cooldown:
                return 0


        self.event_history[event] = current_time


        points = scores.get(event, 0)

        self.score += points


        return points


    def is_calibrating(self, now=None):
        """True while the scene is still being learned; don't judge yet.

        The static filter needs a couple of seconds to work out which parts of
        the frame are furniture, and until it has, background clutter WILL be
        reported. Scoring during that window guarantees a false "Phone detected"
        as the first thing anyone sees.
        """
        current_time = now if now is not None else time.time()
        return current_time - self.started_at < STARTUP_GRACE_SECONDS


    def decay(self, now=None):
        """Bleed the score back down while nothing is being flagged.

        Without this the score only ever climbs, so a SINGLE false positive --
        the two-second warm-up before the static filter learns the room, say --
        pins the session at "Suspicious" for as long as it runs, long after the
        thing that caused it is gone. A score that cannot fall is not measuring
        current behaviour, it is measuring whether anything ever happened.

        Decaying makes it a *recent* suspicion level: sustained behaviour still
        accumulates faster than it drains (an event fires every `cooldown`
        seconds, and every event is worth more than `SCORE_DECAY_PER_SEC` times
        the cooldown), so a genuine phone user still climbs to High Risk and
        stays there. Call once per frame.
        """
        current_time = now if now is not None else time.time()

        elapsed = current_time - self._last_decay
        self._last_decay = current_time

        if elapsed > 0 and self.score > 0:
            self.score = max(0, self.score - SCORE_DECAY_PER_SEC * elapsed)

        return self.score



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


        # Looking away is TWO events, because it rests on two signals of very
        # different quality and collapsing them would launder a guess into
        # looking like a measurement:
        #
        #   "Looking away"     head turned past the CALIBRATED 30 deg (f1 0.869)
        #   "Gaze off screen"  head forward, but the eyes are off-centre -- from
        #                      the UNCALIBRATED 0.35/0.65 iris ratios
        #
        # Gaze earns its place: head yaw only catches a turned HEAD, so glancing
        # at notes beside the screen is invisible to it. It scores less because
        # nobody has checked those cut-points against ground truth.
        #
        # abs() because the yaw SIGN is unreliable (RQDecomp3x3 flip).
        off_centre_gaze = gaze is not None and gaze != "CENTER"

        if head_yaw is not None:
            if abs(head_yaw) >= HEAD_YAW_LOOKING_AWAY:
                events.append("Looking away")
            elif off_centre_gaze:
                # Head forward, eyes elsewhere -- exactly what yaw cannot see.
                events.append("Gaze off screen")

        elif off_centre_gaze:
            # No head pose this frame, so gaze is all we have. Still the weaker
            # event: falling back to a guess does not make it a measurement.
            events.append("Gaze off screen")


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