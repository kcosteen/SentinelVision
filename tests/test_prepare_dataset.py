"""
Unit tests for the Phase 2 dataset assembly.

The property under test is **no leakage**: frames cut from one clip are
near-duplicates, so if any clip appears on both sides of the train/val split the
reported mAP is inflated and meaningless. Same rule as the grouped split in
the Phase 1 windowing code, tested for the same reason.
"""

from src.detection.prepare_dataset import (
    CLASSES,
    cap_backgrounds,
    clip_of,
    remap_label_lines,
    split_own_by_clip,
    split_public,
)


def make_items(clip_frame_counts):
    """[(image_path, label_lines, clip_id)] for {clip: n_frames}."""
    items = []
    for clip, count in clip_frame_counts.items():
        for index in range(count):
            items.append((f"{clip}_f{index:05d}.jpg", ["0 0.5 0.5 0.2 0.2"], clip))
    return items


def test_clip_of_recovers_the_source_clip():
    assert clip_of("phone_001_f00030.jpg") == "phone_001"
    assert clip_of("data/frames/phone_012_f00005.jpg") == "phone_012"


def test_clip_of_handles_a_name_without_a_frame_marker():
    # Not a frame we cut -- fall back to the stem rather than crashing.
    assert clip_of("some_image.jpg") == "some_image"


def test_clip_of_keeps_multiword_behaviour_names():
    assert clip_of("looking_away_003_f00120.jpg") == "looking_away_003"


def test_no_clip_appears_in_both_splits():
    items = make_items({f"phone_{i:03d}": 20 for i in range(1, 11)})
    train, val, val_clips = split_own_by_clip(items, val_fraction=0.2, seed=42)

    train_clips = {clip_of(path) for path, _ in train}
    val_clip_set = {clip_of(path) for path, _ in val}
    assert train_clips.isdisjoint(val_clip_set)
    assert set(val_clips) == val_clip_set


def test_every_frame_lands_somewhere():
    items = make_items({"phone_001": 5, "phone_002": 7, "phone_003": 3})
    train, val, _ = split_own_by_clip(items, val_fraction=0.34, seed=1)
    assert len(train) + len(val) == len(items)


def test_split_is_deterministic_for_a_given_seed():
    items = make_items({f"phone_{i:03d}": 4 for i in range(1, 9)})
    first = split_own_by_clip(items, val_fraction=0.25, seed=7)[2]
    second = split_own_by_clip(items, val_fraction=0.25, seed=7)[2]
    assert first == second


def test_a_single_clip_is_never_split_into_an_empty_train_set():
    """With one clip there is nothing to hold out -- keep it all in train."""
    items = make_items({"phone_001": 10})
    train, val, val_clips = split_own_by_clip(items, val_fraction=0.2, seed=42)
    assert len(train) == 10
    assert val == [] and val_clips == []


def test_at_least_one_clip_is_held_out_when_more_than_one_exists():
    # 2 clips * 0.2 rounds to 0, but a val split of nothing is useless.
    items = make_items({"phone_001": 5, "phone_002": 5})
    _, val, val_clips = split_own_by_clip(items, val_fraction=0.2, seed=42)
    assert len(val_clips) == 1
    assert len(val) == 5


def test_empty_input_does_not_crash():
    train, val, val_clips = split_own_by_clip([], val_fraction=0.2, seed=42)
    assert (train, val, val_clips) == ([], [], [])


def test_public_split_respects_the_requested_fraction():
    items = [(f"img_{i}.jpg", ["0 0.5 0.5 0.1 0.1"]) for i in range(100)]
    train, val = split_public(items, val_fraction=0.2, seed=42)
    assert len(train) == 80 and len(val) == 20
    # Nothing duplicated or dropped.
    assert len({p for p, _ in train} | {p for p, _ in val}) == 100


def test_single_class_dataset():
    """The converter writes class id 0; that must be the only class declared."""
    assert CLASSES == ["phone"]


# --- Roboflow export ingestion -------------------------------------------------
#
# The remapping is the highest-risk code in the pipeline: get it wrong and every
# 'person' box silently becomes a 'phone', which would poison training without
# raising anything.


def test_remap_rewrites_the_upstream_index_to_ours():
    """Upstream 'phone' at index 2 must come out as our class 0."""
    lines = ["2 0.5 0.5 0.2 0.2"]
    kept, dropped = remap_label_lines(lines, ["calculator", "person", "phone"])
    assert kept == ["0 0.5 0.5 0.2 0.2"]
    assert dropped == 0


def test_remap_drops_classes_we_do_not_train():
    """person / calculator / face must be dropped, never folded into class 0."""
    lines = [
        "0 0.1 0.1 0.1 0.1",   # calculator
        "1 0.2 0.2 0.1 0.1",   # person
        "2 0.3 0.3 0.1 0.1",   # phone
    ]
    kept, dropped = remap_label_lines(lines, ["calculator", "person", "phone"])
    assert kept == ["0 0.3 0.3 0.1 0.1"]
    assert dropped == 2


def test_remap_is_case_and_whitespace_insensitive():
    kept, _ = remap_label_lines(["0 0.5 0.5 0.2 0.2"], [" Mobile Phone "])
    assert kept == ["0 0.5 0.5 0.2 0.2"]


def test_remap_of_an_all_unknown_image_yields_a_background():
    kept, dropped = remap_label_lines(["0 0.5 0.5 0.2 0.2"], ["person"])
    assert kept == []
    assert dropped == 1


def test_remap_ignores_malformed_lines():
    kept, _ = remap_label_lines(["", "2", "not a number 1 2 3 4"], ["a", "b", "phone"])
    assert kept == []


def test_remap_keeps_only_the_five_yolo_fields():
    """Roboflow sometimes appends a confidence column; it must not survive."""
    kept, _ = remap_label_lines(["0 0.5 0.5 0.2 0.2 0.99"], ["phone"])
    assert kept == ["0 0.5 0.5 0.2 0.2"]


def test_backgrounds_are_capped_relative_to_positives():
    positives = [(f"p{i}.jpg", ["0 0.5 0.5 0.1 0.1"]) for i in range(100)]
    backgrounds = [(f"b{i}.jpg", []) for i in range(500)]
    kept = cap_backgrounds(positives, backgrounds, ratio=0.1, seed=42)
    assert len(kept) == 10


def test_backgrounds_are_kept_whole_when_already_under_the_cap():
    positives = [(f"p{i}.jpg", ["0 0.5 0.5 0.1 0.1"]) for i in range(100)]
    backgrounds = [(f"b{i}.jpg", []) for i in range(5)]
    assert cap_backgrounds(positives, backgrounds, ratio=0.1, seed=42) == backgrounds


def test_no_positives_means_no_backgrounds_survive():
    """Without positives the cap is 0 -- a background-only dataset is useless."""
    assert cap_backgrounds([], [("b.jpg", [])], ratio=0.1, seed=42) == []
