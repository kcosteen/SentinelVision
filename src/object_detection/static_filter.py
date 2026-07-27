"""Suppress detections that never move -- they are furniture, not behaviour.

The fine-tuned detector calls the wall shelf behind this desk a `cell phone` at
0.60-0.71 on a live webcam. A phone actually being used scores 0.72-0.79. There
is no confidence threshold between those, so no amount of tuning separates them:
the two distributions genuinely overlap.

What DOES separate them is motion. A shelf is bolted to the wall and lands in the
same pixels frame after frame; a phone someone is using moves, and a phone being
picked up appears somewhere new. That signal was being thrown away by scoring
every frame independently.

So: track each detection box across frames, and once one has sat in essentially
the same place for `static_after` frames, treat it as background and stop
reporting it. Nothing is configured per-room -- the filter learns whatever
happens to be nailed down in front of *this* camera.

**Applied to `cell phone` only, by default.** The rule "if it isn't moving it
isn't behaviour" is true for a phone in use; it is not true for a book lying open
on the desk, and it is emphatically not true for `person` -- a candidate sitting
still is the normal case, and suppressing them would break the whole app.

**Known tradeoff:** a phone left motionless on the desk for two seconds stops
being reported until it moves again. That is the deliberate cost of the rule.
Detection resumes the moment it is picked up, which is the moment that matters.

**Warm-up:** for the first `static_after` frames after start-up nothing has a
history yet, so background objects ARE reported briefly before settling. Honest
behaviour for a filter that learns the scene online, and it clears in ~2s.

Pure geometry and counters -- no OpenCV, no model -- so it unit-tests without a
camera.
"""

# Boxes that overlap this much are treated as "the same thing, still there".
# Loose enough to absorb the pixel jitter a detector produces on a static object,
# tight enough that a phone moving a few centimetres reads as a new position.
DEFAULT_IOU_MATCH = 0.80

# Frames in the same place before something is called background. ~2s at 30fps.
DEFAULT_STATIC_AFTER = 60

# Frames a track survives without being seen, so a brief miss doesn't reset a
# streak that took two seconds to build.
DEFAULT_FORGET_AFTER = 15

# Only classes where "not moving" genuinely implies "not behaviour".
DEFAULT_LABELS = frozenset({"cell phone"})


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    overlap = iw * ih
    if overlap <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


class StaticRegionFilter:
    """Learns which detections belong to the scene rather than to the candidate.

        f = StaticRegionFilter()
        keep = f.step([("cell phone", (10, 10, 50, 50))])   # [True] while new
        ...                                                  # after ~2s
        keep = f.step([("cell phone", (10, 10, 50, 50))])   # [False] -- furniture
    """

    def __init__(self, iou_match=DEFAULT_IOU_MATCH,
                 static_after=DEFAULT_STATIC_AFTER,
                 forget_after=DEFAULT_FORGET_AFTER,
                 labels=DEFAULT_LABELS):
        self.iou_match = iou_match
        self.static_after = static_after
        self.forget_after = forget_after
        self.labels = frozenset(labels)
        self._tracks = []
        self._frame = 0

    def step(self, observations):
        """One frame of (label, box) pairs -> a keep/suppress flag for each.

        Returns a list of booleans parallel to `observations`: True to report the
        detection, False to suppress it as background.
        """
        self._frame += 1
        keep = []
        matched = set()

        for label, box in observations:
            if label not in self.labels:
                keep.append(True)          # not a class this filter governs
                continue

            track = self._match(label, box, matched)
            if track is None:
                track = {"label": label, "box": box, "streak": 0}
                self._tracks.append(track)

            track["streak"] += 1
            track["box"] = box             # follow the small jitter
            track["last_seen"] = self._frame
            matched.add(id(track))

            keep.append(track["streak"] < self.static_after)

        self._prune()
        return keep

    def _match(self, label, box, already_matched):
        """The best still-unclaimed track for this box, or None."""
        best, best_iou = None, self.iou_match
        for track in self._tracks:
            if track["label"] != label or id(track) in already_matched:
                continue
            score = iou(track["box"], box)
            if score >= best_iou:
                best, best_iou = track, score
        return best

    def _prune(self):
        """Drop tracks that haven't been seen recently.

        Without this, a phone that rests somewhere briefly would keep its streak
        forever and stay suppressed after it had plainly moved on.
        """
        self._tracks = [
            t for t in self._tracks
            if self._frame - t.get("last_seen", self._frame) <= self.forget_after
        ]

    @property
    def suppressed_count(self):
        """How many regions are currently classed as background (for the HUD)."""
        return sum(1 for t in self._tracks if t["streak"] >= self.static_after)
