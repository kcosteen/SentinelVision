"""Unit tests for background (static-object) suppression.

The real case these encode: a wall shelf that the fine-tuned detector calls a
`cell phone` at 0.60-0.71, overlapping a real phone's 0.72-0.79 so completely
that no confidence threshold separates them. Motion does. These tests pin that
the filter forgets furniture, notices a phone the moment it moves, and never
touches classes it does not govern.
"""

from src.object_detection.static_filter import StaticRegionFilter, iou

SHELF = (890.0, 120.0, 1010.0, 400.0)
PHONE = (300.0, 250.0, 420.0, 530.0)


def run(filter_, box, frames, label="cell phone"):
    """Feed the same box for N frames; return the keep-flag from the last one."""
    keep = [True]
    for _ in range(frames):
        keep = filter_.step([(label, box)])
    return keep[0]


# --- iou -------------------------------------------------------------------

def test_iou_identical_boxes():
    assert iou(SHELF, SHELF) == 1.0


def test_iou_disjoint_boxes():
    assert iou(SHELF, PHONE) == 0.0


def test_iou_partial_overlap():
    # Two unit squares offset by half -> intersection 0.5, union 1.5.
    assert iou((0, 0, 1, 1), (0.5, 0, 1.5, 1)) == 1 / 3


def test_iou_handles_degenerate_box():
    assert iou((0, 0, 0, 0), (0, 0, 1, 1)) == 0.0


# --- the core behaviour ----------------------------------------------------

def test_static_object_is_reported_then_suppressed():
    """Furniture is visible during warm-up, then learned and dropped."""
    f = StaticRegionFilter(static_after=10)
    assert run(f, SHELF, 9) is True, "should still report before the threshold"
    assert run(f, SHELF, 1) is False, "should be background by frame 10"
    assert run(f, SHELF, 50) is False, "and stay background"


def test_moving_object_is_never_suppressed():
    """A phone drifting across the frame never builds a streak."""
    f = StaticRegionFilter(static_after=10)
    for i in range(40):
        box = (300.0 + i * 25, 250.0, 420.0 + i * 25, 530.0)
        assert f.step([("cell phone", box)])[0] is True


def test_phone_appearing_elsewhere_is_reported_despite_learned_shelf():
    """The whole point: suppressing the shelf must not blind us to a real phone."""
    f = StaticRegionFilter(static_after=10)
    run(f, SHELF, 20)                       # shelf now background

    keep = f.step([("cell phone", SHELF), ("cell phone", PHONE)])
    assert keep == [False, True]


def test_picked_up_phone_is_reported_again():
    """A phone that rested, then moved, must come back immediately."""
    f = StaticRegionFilter(static_after=10)
    assert run(f, PHONE, 20) is False       # rested long enough to be scenery

    moved = (600.0, 250.0, 720.0, 530.0)    # picked up -> new location
    assert f.step([("cell phone", moved)])[0] is True


def test_small_jitter_does_not_reset_the_streak():
    """Detector noise on a static object must not read as motion."""
    f = StaticRegionFilter(static_after=10)
    for i in range(20):
        nudge = 1.0 if i % 2 else -1.0      # a pixel of wobble
        box = (SHELF[0] + nudge, SHELF[1], SHELF[2] + nudge, SHELF[3])
        keep = f.step([("cell phone", box)])[0]
    assert keep is False


def test_track_is_forgotten_after_absence():
    """Gone long enough, then back -> a fresh streak, so it is reported again."""
    f = StaticRegionFilter(static_after=10, forget_after=5)
    assert run(f, SHELF, 20) is False

    for _ in range(6):                      # absent past forget_after
        f.step([])

    assert f.step([("cell phone", SHELF)])[0] is True


def test_brief_miss_does_not_reset_the_streak():
    """One dropped frame must not undo a streak that took 2s to build."""
    f = StaticRegionFilter(static_after=10, forget_after=5)
    run(f, SHELF, 20)
    f.step([])                              # single missed frame
    assert f.step([("cell phone", SHELF)])[0] is False


# --- scope -----------------------------------------------------------------

def test_person_is_never_suppressed():
    """A candidate sitting still is the NORMAL case -- suppressing them would
    break the whole app, so `person` must not be governed by this filter."""
    f = StaticRegionFilter(static_after=10)
    for _ in range(100):
        assert f.step([("person", (100.0, 100.0, 500.0, 700.0))])[0] is True


def test_book_is_not_suppressed_by_default():
    """A book lying open on the desk is static AND genuinely present."""
    f = StaticRegionFilter(static_after=10)
    assert run(f, SHELF, 50, label="book") is True


def test_two_static_objects_tracked_independently():
    f = StaticRegionFilter(static_after=10)
    for _ in range(20):
        keep = f.step([("cell phone", SHELF), ("cell phone", PHONE)])
    assert keep == [False, False]
    assert f.suppressed_count == 2
