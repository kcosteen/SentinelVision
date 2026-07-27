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

# MEASURED on the live webcam this was built for: over 150 frames of the static
# shelf, consecutive-frame IoU cleared 0.80 only **12%** of the time, while the
# box CENTRE drifted just 8px (p90) on a 1280-wide frame -- 0.6% of the width.
#
# The box is not moving; its SIZE is thrashing, because the detector brackets the
# whole shelf one frame and just the keyboard on it the next. IoU collapses under
# that (a box 25% larger about the same centre already falls below 0.80), so an
# IoU-matched filter sees a brand-new object every frame and never builds a
# streak. That is exactly how the first version of this failed.
#
# So association and staticness both work off the CENTRE, in units of the box's
# own size, which keeps it resolution-independent.

# How close two centres must be to count as the same tracked object. Generous:
# it only has to keep following one thing between frames.
DEFAULT_ASSOC_RATIO = 0.50

# How far a centre may sit from where its streak began and still count as "hasn't
# moved". Tight -- measured jitter is ~0.04 of box size, so 0.15 absorbs it with
# room to spare while a phone that actually travels re-anchors within a frame or
# two and never accumulates a streak.
DEFAULT_STATIC_RATIO = 0.15

# Frames in the same place before something is called background. ~2s at 30fps.
DEFAULT_STATIC_AFTER = 60

# Frames a track survives without being seen, so a brief miss doesn't reset a
# streak that took two seconds to build.
DEFAULT_FORGET_AFTER = 15

# Only classes where "not moving" genuinely implies "not behaviour".
DEFAULT_LABELS = frozenset({"cell phone"})


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes.

    Kept for diagnostics. NOT used for matching -- see the note above on why it
    is the wrong measure when a detector's box size is unstable.
    """
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


def centre(box):
    """(cx, cy) of an (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_scale(box):
    """A single size for a box: the mean of its width and height.

    Used to express distances as a fraction of the object's own size, so the
    same ratios work at any resolution and for any object.
    """
    x1, y1, x2, y2 = box
    return (abs(x2 - x1) + abs(y2 - y1)) / 2.0


def centre_offset(a, b):
    """Distance between two box centres, as a fraction of their mean size.

    0.0 means concentric; 1.0 means a whole box-width apart.
    """
    (ax, ay), (bx, by) = centre(a), centre(b)
    scale = (box_scale(a) + box_scale(b)) / 2.0
    if scale <= 0:
        return float("inf")
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 / scale


class StaticRegionFilter:
    """Learns which detections belong to the scene rather than to the candidate.

        f = StaticRegionFilter()
        keep = f.step([("cell phone", (10, 10, 50, 50))])   # [True] while new
        ...                                                  # after ~2s
        keep = f.step([("cell phone", (10, 10, 50, 50))])   # [False] -- furniture
    """

    def __init__(self, assoc_ratio=DEFAULT_ASSOC_RATIO,
                 static_ratio=DEFAULT_STATIC_RATIO,
                 static_after=DEFAULT_STATIC_AFTER,
                 forget_after=DEFAULT_FORGET_AFTER,
                 labels=DEFAULT_LABELS):
        self.assoc_ratio = assoc_ratio
        self.static_ratio = static_ratio
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
                track = {"label": label, "box": box, "anchor": box, "streak": 0}
                self._tracks.append(track)

            # Staticness is measured against where the streak BEGAN, not against
            # the previous frame. Frame-to-frame comparison would call a slow
            # drift static: each step is tiny, so the streak would keep building
            # while the object crossed the whole frame. Anchoring means anything
            # that actually travels re-anchors and starts counting from zero.
            if centre_offset(box, track["anchor"]) <= self.static_ratio:
                track["streak"] += 1
            else:
                track["anchor"] = box
                track["streak"] = 1

            track["box"] = box             # follow it for association
            track["last_seen"] = self._frame
            matched.add(id(track))

            keep.append(track["streak"] < self.static_after)

        self._prune()
        return keep

    def _match(self, label, box, already_matched):
        """The nearest still-unclaimed track for this box, or None.

        Matched on centre proximity rather than IoU: the detector's box size is
        unstable on static clutter, and IoU cannot tell that apart from motion.
        """
        best, best_offset = None, self.assoc_ratio
        for track in self._tracks:
            if track["label"] != label or id(track) in already_matched:
                continue
            offset = centre_offset(track["box"], box)
            if offset <= best_offset:
                best, best_offset = track, offset
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
