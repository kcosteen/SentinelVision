"""Unit tests for background (static-object) suppression.

The real case these encode: a wall shelf the fine-tuned detector calls a
`cell phone` at 0.60-0.71, overlapping a real phone's 0.72-0.79 so completely
that no confidence threshold separates them (confirmed by a blank-background
control, where the false detections vanish).

Two earlier designs failed on the same fragility -- both tried to follow an
individual box across frames, and this detector does not produce a followable
box on clutter. These tests pin the properties that made those versions fail, so
the occupancy-grid replacement cannot regress into them:

* box size may thrash freely (only the centre is used)
* several competing boxes may appear at once (no identities to confuse)
* the frame rate may be anything (thresholds are wall-clock seconds)
"""

from src.object_detection.static_filter import (
    StaticRegionFilter,
    centre,
    centre_inside,
    held_by_person,
    iou,
)

FRAME = (1280, 720)

SHELF = (890.0, 120.0, 1010.0, 400.0)     # background clutter, right of frame
PHONE = (300.0, 250.0, 420.0, 530.0)      # a phone held up, well away from it


def soak(f, box, seconds, start=0.0, hz=10.0, label="cell phone"):
    """Feed one box steadily for `seconds`; return the final keep flag."""
    keep, t, step = [True], start, 1.0 / hz
    n = int(seconds * hz)
    for i in range(n):
        t = start + i * step
        keep = f.step([(label, box)], FRAME, now=t)
    return keep[0], t


# --- helpers ---------------------------------------------------------------

def test_centre_of_a_box():
    assert centre((0.0, 0.0, 10.0, 20.0)) == (5.0, 10.0)


def test_iou_still_available_for_diagnostics():
    assert iou(SHELF, SHELF) == 1.0
    assert iou(SHELF, PHONE) == 0.0


# --- core behaviour --------------------------------------------------------

def test_static_object_is_reported_then_suppressed():
    f = StaticRegionFilter(static_seconds=2.0)
    keep, t = soak(f, SHELF, 1.0)
    assert keep is True, "still learning during warm-up"

    keep, t = soak(f, SHELF, 3.0, start=t)
    assert keep is False, "scenery once the cell has been busy 2s"


def test_moving_object_is_never_suppressed():
    """A phone crossing the frame keeps entering fresh cells."""
    f = StaticRegionFilter(static_seconds=2.0)
    for i in range(100):
        box = (300.0 + i * 9, 250.0, 420.0 + i * 9, 530.0)
        assert f.step([("cell phone", box)], FRAME, now=i * 0.1)[0] is True


def test_phone_elsewhere_is_reported_despite_learned_shelf():
    """The whole point: suppressing the shelf must not blind us to a real phone."""
    f = StaticRegionFilter(static_seconds=2.0)
    _, t = soak(f, SHELF, 3.0)

    keep = f.step([("cell phone", SHELF), ("cell phone", PHONE)], FRAME, now=t + 0.1)
    assert keep == [False, True]


def test_picked_up_phone_is_reported_again():
    f = StaticRegionFilter(static_seconds=2.0)
    keep, t = soak(f, PHONE, 3.0)
    assert keep is False, "rested long enough to read as scenery"

    moved = (700.0, 250.0, 820.0, 530.0)
    assert f.step([("cell phone", moved)], FRAME, now=t + 0.1)[0] is True


# --- the failures that killed the two previous designs ---------------------

def test_wildly_thrashing_box_size_still_suppresses():
    """Killed design #1. Consecutive-frame IoU measured 0.80 only 12% of the
    time because the box size swings; suppression must not care."""
    f = StaticRegionFilter(static_seconds=2.0)
    keep = [True]
    for i in range(60):
        grow = 80.0 if i % 2 else 0.0
        box = (SHELF[0] - grow, SHELF[1] - grow, SHELF[2] + grow, SHELF[3] + grow)
        keep = f.step([("cell phone", box)], FRAME, now=i * 0.1)
    assert keep[0] is False

    # ...and those boxes really would have defeated IoU matching.
    big = (SHELF[0] - 80, SHELF[1] - 80, SHELF[2] + 80, SHELF[3] + 80)
    assert iou(SHELF, big) < 0.80


def test_many_competing_boxes_still_suppresses():
    """Killed design #2. The shelf emits several boxes that split and swap;
    with no identities to confuse, each simply marks its own cell."""
    f = StaticRegionFilter(static_seconds=2.0)
    spread = [
        SHELF,
        (900.0, 130.0, 1000.0, 300.0),
        (880.0, 200.0, 1020.0, 410.0),
    ]
    keep = [True]
    for i in range(60):
        boxes = spread if i % 2 else list(reversed(spread))   # order swaps too
        keep = f.step([("cell phone", b) for b in boxes], FRAME, now=i * 0.1)
    assert keep == [False, False, False]


def test_threshold_is_wall_clock_not_frames():
    """A slow machine must behave like a fast one. 2s of footage is 2s whether
    it arrived as 60 frames or 6."""
    fast = StaticRegionFilter(static_seconds=2.0)
    slow = StaticRegionFilter(static_seconds=2.0)

    assert soak(fast, SHELF, 3.0, hz=30.0)[0] is False
    assert soak(slow, SHELF, 3.0, hz=3.0)[0] is False


def test_centre_jitter_across_a_cell_boundary_still_matures():
    """A centre sitting on a boundary must not flip between two cells forever."""
    f = StaticRegionFilter(static_seconds=2.0)
    cell = max(FRAME) * f.cell_ratio
    x = cell * 5                                   # exactly on an edge
    keep = [True]
    for i in range(60):
        nudge = 3.0 if i % 2 else -3.0
        box = (x + nudge - 60, 300.0, x + nudge + 60, 560.0)
        keep = f.step([("cell phone", box)], FRAME, now=i * 0.1)
    assert keep[0] is False


# --- forgetting ------------------------------------------------------------

def test_occlusion_does_not_make_the_filter_relearn():
    """Reported from the real camera: sitting forward occludes the shelf, so
    nothing is detected there; leaning back reveals it again and it flagged EVERY
    time. A 1s forget window was wiping everything already learned. Furniture
    does not stop being furniture because somebody leaned in front of it."""
    f = StaticRegionFilter(static_seconds=2.0, forget_seconds=1.0,
                           remember_seconds=60.0)
    keep, t = soak(f, SHELF, 3.0)
    assert keep is False, "learned while leaning back"

    # Lean forward for 12s -- shelf fully occluded, no detections at all.
    for i in range(120):
        f.step([], FRAME, now=t + i * 0.1)

    # Lean back: it must be remembered, not relearned.
    assert f.step([("cell phone", SHELF)], FRAME, now=t + 12.5)[0] is False


def test_repeated_lean_cycles_never_flag():
    """The actual reported symptom, repeated."""
    f = StaticRegionFilter(static_seconds=2.0)
    keep, t = soak(f, SHELF, 3.0)
    assert keep is False

    for cycle in range(5):
        for i in range(50):                      # ~5s occluded
            f.step([], FRAME, now=t + i * 0.1)
        t += 5.0
        assert f.step([("cell phone", SHELF)], FRAME, now=t)[0] is False, cycle


def test_shifted_box_centre_still_reads_as_known_scenery():
    """Leaning back reveals more of the shelf, moving the detector's box centre
    by more than one cell. Confirmed scenery has to cover that."""
    f = StaticRegionFilter(static_seconds=2.0)
    keep, t = soak(f, SHELF, 3.0)
    assert keep is False

    cell = max(FRAME) * f.cell_ratio
    shifted = (SHELF[0] - cell, SHELF[1], SHELF[2] + cell, SHELF[3])
    assert f.step([("cell phone", shifted)], FRAME, now=t + 0.2)[0] is False


def test_region_is_forgotten_after_a_very_long_absence():
    """Memory is long, not infinite -- move the furniture and it re-evaluates."""
    f = StaticRegionFilter(static_seconds=2.0, remember_seconds=60.0)
    keep, t = soak(f, SHELF, 3.0)
    assert keep is False

    assert f.step([("cell phone", SHELF)], FRAME, now=t + 120.0)[0] is True


def test_a_dropped_frame_does_not_undo_progress():
    f = StaticRegionFilter(static_seconds=2.0, forget_seconds=1.0)
    keep, t = soak(f, SHELF, 3.0)
    assert keep is False
    assert f.step([("cell phone", SHELF)], FRAME, now=t + 0.3)[0] is False


# --- scope -----------------------------------------------------------------

def test_person_is_never_suppressed():
    """A candidate sitting still is the NORMAL case; suppressing them would
    break the entire app."""
    f = StaticRegionFilter(static_seconds=2.0)
    for i in range(100):
        keep = f.step([("person", (100.0, 100.0, 500.0, 700.0))], FRAME, now=i * 0.1)
        assert keep[0] is True


def test_book_is_not_suppressed_by_default():
    """A book lying open on a desk is static AND genuinely present."""
    f = StaticRegionFilter(static_seconds=2.0)
    assert soak(f, SHELF, 5.0, label="book")[0] is True


def test_empty_frame_is_handled():
    f = StaticRegionFilter()
    assert f.step([], FRAME, now=0.0) == []


# --- the person exemption --------------------------------------------------
#
# "A phone in use moves" is FALSE for the case that matters most: a phone being
# READ is held still. Static suppression alone would learn it as furniture.

PERSON = (350.0, 60.0, 900.0, 720.0)      # candidate, centre-left of frame


def test_centre_inside_basic():
    assert centre_inside((400.0, 300.0, 500.0, 400.0), PERSON) is True
    assert centre_inside(SHELF, PERSON) is False


def test_centre_inside_margin_extends_the_box():
    just_outside = (920.0, 300.0, 960.0, 340.0)
    assert centre_inside(just_outside, PERSON, margin=0.0) is False
    assert centre_inside(just_outside, PERSON, margin=0.25) is True


def test_default_margin_is_strict_enough_to_exclude_adjacent_clutter():
    """Why margin defaults to 0. The person box runs head-to-floor, so a 25%
    margin grows it by 130+ px and reaches the shelf sitting right beside them --
    which would re-admit the exact false positive the filter exists to remove."""
    assert held_by_person(SHELF, [PERSON]) is False           # default, strict
    assert held_by_person(SHELF, [PERSON], margin=0.25) is True   # the trap


def test_phone_on_the_person_is_held():
    assert held_by_person((500.0, 300.0, 620.0, 560.0), [PERSON]) is True


def test_shelf_is_not_held_by_the_person():
    assert held_by_person(SHELF, [PERSON]) is False


def test_held_by_person_with_nobody_detected():
    assert held_by_person(SHELF, []) is False


def test_a_phone_read_completely_still_is_still_reported():
    """THE regression test for this bug. A phone held rock-steady in front of the
    candidate for 10 seconds -- someone reading it -- must never be suppressed,
    even though the filter has every reason to call that region static."""
    f = StaticRegionFilter(static_seconds=2.0)
    reading = (500.0, 300.0, 620.0, 560.0)

    suppressed_at_some_point = False
    for i in range(100):
        keep = f.step([("cell phone", reading)], FRAME, now=i * 0.1)[0]
        if not keep and not held_by_person(reading, [PERSON]):
            suppressed_at_some_point = True

    # The grid alone WOULD have learned it -- that is the trap.
    assert f.step([("cell phone", reading)], FRAME, now=10.1)[0] is False
    # ...but the person exemption rescues it.
    assert held_by_person(reading, [PERSON]) is True
    assert suppressed_at_some_point is False
