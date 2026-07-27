"""Suppress detections that never move -- they are furniture, not behaviour.

The fine-tuned detector calls this desk's wall shelf a `cell phone` at 0.60-0.71
on a live webcam. A phone actually in use scores 0.72-0.79. Those ranges touch,
so no confidence threshold separates them -- confirmed by a blank-background
control run, where the false detections vanish entirely.

What separates them is motion: a shelf is bolted to the wall, a phone in use is
not. The problem is measuring "hasn't moved" reliably.

**Two earlier designs failed, both for the same reason.** Both tried to follow an
individual box from frame to frame and time how long it stayed put:

* Matching by IoU >= 0.80 -- measured on the live camera, consecutive frames
  matched only **12%** of the time. The detector brackets the whole shelf one
  frame and just the keyboard on it the next, and IoU collapses under that size
  swing, so every frame looked like a brand-new object.
* Matching by centre proximity -- better, but still assumes the shelf produces
  ONE coherent box to follow. It produces several competing ones that split,
  merge and swap, so associating them across frames stays unreliable.

The fragile part was never the threshold, it was the **association**: deciding
which box this frame corresponds to which box last frame. So this version does
not do association at all.

**Instead: a spatial occupancy grid.** The frame is divided into coarse cells.
Every phone detection marks the cell its centre falls in. A cell that has been
continuously occupied for `static_seconds` is scenery, and detections centred
there are dropped. Nothing needs to be tracked, matched, or identified.

That is immune to all three things that broke the earlier attempts: box size can
thrash freely (only the centre matters), the shelf can emit ten boxes at once
(each just marks its own cell), and boxes can swap identity every frame (there
are no identities).

**Time, not frames.** The threshold is wall-clock seconds. This pipeline runs
YOLO plus two MediaPipe models per frame, so its frame rate depends entirely on
the machine -- "60 frames" silently means 2s on one computer and 10s on another.
Seconds mean the same thing everywhere.

**Applied to `cell phone` only, by default.** "If it isn't moving it isn't
behaviour" holds for a phone in use. It does not hold for a book lying open on a
desk, and it emphatically does not hold for `person` -- a candidate sitting still
is the normal case, and suppressing them would break the whole app.

**Known tradeoffs**, both covered by tests:
* A phone left motionless in one spot for `static_seconds` stops being reported
  until it moves. Detection resumes when it is picked up, which is the moment
  that matters.
* For the first `static_seconds` after start-up, nothing has history yet, so
  background objects ARE reported briefly before the filter settles.

Pure geometry, dictionaries and timestamps -- no OpenCV, no model -- so it
unit-tests without a camera.
"""

import time

# Cell edge as a fraction of the frame's larger side. At 1280x720 this is ~64px.
# Comfortably wider than the ~8px (p90) centre jitter measured on the real
# camera, so a static object stays inside one cell instead of flickering between
# two; small enough that a phone held up in front of the face lands well clear of
# the shelf's cells.
DEFAULT_CELL_RATIO = 0.05

# Seconds a cell must stay occupied before it counts as scenery.
DEFAULT_STATIC_SECONDS = 2.0

# Seconds an UNCONFIRMED cell survives unoccupied. Short on purpose: a phone
# drifting through a cell must not leave credit behind that later matures.
DEFAULT_FORGET_SECONDS = 1.0

# Seconds a CONFIRMED region is remembered after it was last seen. Long on
# purpose, and the fix for a bug found on the real camera: sitting forward, the
# user's head and shoulders occlude the shelf, so nothing is detected there and a
# 1s forget window wiped everything the filter had learned. Leaning back revealed
# the shelf into cells with no history, which warmed up from scratch and flagged
# every single time.
#
# "I have not seen it lately" is not "it is gone". Furniture does not stop being
# furniture because somebody leaned in front of it, so once a region has proven
# itself static it stays known through occlusion, re-framing and brief absence.
DEFAULT_REMEMBER_SECONDS = 60.0

# How many cells out to look when deciding whether a detection sits in known
# scenery. The visible extent of a large object changes as the user moves --
# leaning back reveals more shelf, which shifts the detector's box centre by more
# than one cell -- so confirmed scenery casts a slightly wider shadow than the
# single cell it was learned in. Still far narrower than the gap between the
# shelf and where a phone is held up in front of the face.
DEFAULT_NEIGHBOURHOOD = 2

# Only classes where "not moving" genuinely implies "not behaviour".
DEFAULT_LABELS = frozenset({"cell phone"})


def centre(box):
    """(cx, cy) of an (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def centre_inside(box, container, margin=0.0):
    """Is `box`'s centre inside `container`, optionally grown by a margin?

    `margin` is a fraction of the container's own size, so a person detected
    close to the camera gets a proportionally larger allowance than one sitting
    far back.
    """
    cx, cy = centre(box)
    x1, y1, x2, y2 = container
    pad_x = abs(x2 - x1) * margin
    pad_y = abs(y2 - y1) * margin
    return (x1 - pad_x) <= cx <= (x2 + pad_x) and (y1 - pad_y) <= cy <= (y2 + pad_y)


def held_by_person(box, person_boxes, margin=0.0):
    """True when this object sits on or beside a detected person.

    The escape hatch for static suppression. The premise "a phone in use moves"
    turns out to be FALSE for the case that matters most: a phone being *read* is
    held still, and reading a phone is exactly the behaviour worth catching. Held
    steady for `static_seconds` it would be learned as furniture and ignored.

    What still separates them is position. A phone in use is in someone's hands,
    so its centre falls on the person; a shelf on the wall does not. Anything
    overlapping the candidate is therefore never suppressed, however still it is
    held.

    `margin` defaults to 0 -- strict containment -- and that is deliberate. A
    person box runs from head to the bottom of the frame, so even a 25% margin
    grows it by well over a hundred pixels and swallows clutter sitting just
    beside the candidate. Any margin generous enough to catch a phone held out at
    arm's length is also generous enough to re-admit the shelf, which would undo
    the entire filter. A phone held far enough from the body to fall outside is
    also, conveniently, not being read.
    """
    return any(centre_inside(box, person, margin) for person in person_boxes)


def iou(a, b):
    """Intersection over union of two boxes. Diagnostics only -- see above for
    why this is emphatically NOT what suppression is based on."""
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
    """Learns which parts of the frame are scenery rather than behaviour.

        f = StaticRegionFilter()
        keep = f.step([("cell phone", box)], frame_size=(1280, 720))
        # -> [True] at first, [False] once that spot has been busy for 2s
    """

    def __init__(self, cell_ratio=DEFAULT_CELL_RATIO,
                 static_seconds=DEFAULT_STATIC_SECONDS,
                 forget_seconds=DEFAULT_FORGET_SECONDS,
                 remember_seconds=DEFAULT_REMEMBER_SECONDS,
                 neighbourhood=DEFAULT_NEIGHBOURHOOD,
                 labels=DEFAULT_LABELS):
        self.cell_ratio = cell_ratio
        self.static_seconds = static_seconds
        self.forget_seconds = forget_seconds
        self.remember_seconds = remember_seconds
        self.neighbourhood = neighbourhood
        self.labels = frozenset(labels)
        self._cells = {}

    def step(self, observations, frame_size, now=None):
        """One frame of (label, box) pairs -> a keep/suppress flag for each.

        `frame_size` is (width, height); cell size is derived from it so the same
        ratio behaves identically at any resolution. `now` is injectable so tests
        need no sleeps.
        """
        now = time.time() if now is None else now
        cell_px = max(1.0, max(frame_size) * self.cell_ratio)

        # Mark first, then judge -- so two detections in the same cell on the
        # same frame get the same answer regardless of their order.
        for label, box in observations:
            if label in self.labels:
                self._mark(self._key(label, box, cell_px), now)

        keep = [
            True if label not in self.labels
            else not self._is_scenery(self._key(label, box, cell_px), now)
            for label, box in observations
        ]

        self._prune(now)
        return keep

    def _key(self, label, box, cell_px):
        cx, cy = centre(box)
        return (label, int(cx // cell_px), int(cy // cell_px))

    def _mark(self, key, now):
        """Record that this cell is occupied, extending or restarting its run.

        A run only restarts if the cell was NOT already confirmed. Once a region
        has proven itself static, an occlusion must not demote it back to
        "unknown" and force it to re-earn its status -- that was the lean-back
        bug.
        """
        cell = self._cells.get(key)

        if cell is None or self._is_stale(cell, now):
            cell = {"since": now, "last": now, "confirmed": False}
        else:
            cell["last"] = now                      # continue the run

        if not cell["confirmed"] and now - cell["since"] >= self.static_seconds:
            cell["confirmed"] = True

        self._cells[key] = cell

    def _is_stale(self, cell, now):
        """Past its patience: confirmed scenery gets a much longer leash.

        Confirmed cells expire too, just slowly. Memory that never lapsed would
        mean rearranging the room left permanent blind spots where the detector
        could never report a phone again.
        """
        limit = self.remember_seconds if cell["confirmed"] else self.forget_seconds
        return now - cell["last"] > limit

    def _is_scenery(self, key, now):
        """True when this cell, or one near it, is known scenery.

        Neighbours count for two reasons: a centre resting on a cell boundary
        would otherwise alternate between two cells and neither would mature, and
        a large object's detected extent shifts as the user moves, which walks
        the box centre across cell lines.
        """
        label, ix, iy = key
        reach = range(-self.neighbourhood, self.neighbourhood + 1)

        for dx in reach:
            for dy in reach:
                cell = self._cells.get((label, ix + dx, iy + dy))
                if (cell is not None and cell["confirmed"]
                        and not self._is_stale(cell, now)):
                    return True
        return False

    def _prune(self, now):
        """Forget stale cells -- confirmed scenery on a much longer leash."""
        self._cells = {
            key: cell for key, cell in self._cells.items()
            if not self._is_stale(cell, now)
        }

    @property
    def suppressed_count(self):
        """Regions currently classed as scenery (for the HUD / diagnostics)."""
        return sum(1 for c in self._cells.values() if c["confirmed"])
