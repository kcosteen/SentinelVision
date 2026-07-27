"""Unit tests for name -> class-id resolution.

The bug these guard against is silent: a hard-coded id that doesn't exist in the
loaded model matches nothing, so the pipeline reports "no phone in any frame"
and reads as a broken detector rather than a broken lookup. The COCO and
fine-tune class maps below are the real ones from the two models in this repo.
"""

import pytest

from src.detection.class_ids import (
    normalise_names,
    phone_class_index,
    resolve_class_ids,
)

# The six-class Roboflow proctoring fine-tune.
FINETUNE = {0: "book", 1: "cell phone", 2: "headphone", 3: "laptop", 4: "person", 5: "tv"}

# The COCO ids that matter here, as the 80-class baseline numbers them.
COCO = {0: "person", 63: "laptop", 67: "cell phone", 73: "book"}

# What prepare_dataset.py builds.
SINGLE_CLASS = {0: "phone"}


class _FakeModel:
    """Stands in for an ultralytics model, which exposes `.names`."""

    def __init__(self, names):
        self.names = names


def test_phone_index_differs_per_model():
    """The same object, three different ids -- the whole reason this exists."""
    assert phone_class_index(COCO) == 67
    assert phone_class_index(FINETUNE) == 1
    assert phone_class_index(SINGLE_CLASS) == 0


def test_accepts_model_dict_or_list():
    """Sources in this repo hand us all three shapes."""
    assert phone_class_index(_FakeModel(FINETUNE)) == 1
    assert phone_class_index(FINETUNE) == 1
    # A data.yaml `names:` list, where position is the id.
    assert phone_class_index(["book", "cell phone", "headphone"]) == 1


def test_name_matching_is_case_and_space_insensitive():
    assert phone_class_index({0: "Cell Phone"}) == 0
    assert phone_class_index({0: "  cell phone  "}) == 0
    assert phone_class_index({0: "mobile_phone"}) == 0


def test_missing_phone_class_raises_by_default():
    """Better to stop than to silently measure the wrong object."""
    with pytest.raises(SystemExit):
        phone_class_index({0: "person", 1: "laptop"})


def test_missing_phone_class_returns_none_when_optional():
    assert phone_class_index({0: "person"}, required=False) is None


def test_resolve_reproduces_the_old_hardcoded_coco_list():
    """Regression guard: the COCO path must not change behaviour.

    [0, 63, 67, 73] was hard-coded in feature_extractor.py. Resolving by name
    has to yield exactly that for COCO weights, or this "fix" silently altered
    every baseline number the project has already reported.
    """
    ids = resolve_class_ids(COCO, ["person", "laptop", "cell phone", "book"])
    assert ids == [0, 63, 67, 73]


def test_resolve_maps_the_same_names_onto_finetune_ids():
    ids = resolve_class_ids(FINETUNE, ["person", "laptop", "cell phone", "book"])
    assert ids == [4, 3, 1, 0]


def test_the_bug_this_module_prevents():
    """Those COCO ids against the fine-tune resolve to 'book' and nothing else.

    person_count would log 0 and phone_conf 0.0 on every single frame.
    """
    survivors = [FINETUNE[i] for i in (0, 63, 67, 73) if i in FINETUNE]
    assert survivors == ["book"]


def test_resolve_skips_names_the_model_lacks():
    """The COCO baseline has no 'headphone'. That's a fact, not an error."""
    assert resolve_class_ids(COCO, ["cell phone", "headphone"]) == [67]
    assert resolve_class_ids(FINETUNE, ["cell phone", "headphone"]) == [1, 2]


def test_resolve_preserves_requested_order():
    """Callers log "asked for X, got Y"; order has to line up to compare them."""
    assert resolve_class_ids(FINETUNE, ["tv", "book"]) == [5, 0]


def test_resolve_returns_empty_for_nothing_known():
    assert resolve_class_ids(FINETUNE, ["bicycle", "umbrella"]) == []


def test_normalise_handles_empty_and_none():
    assert normalise_names(None) == {}
    assert normalise_names({}) == {}
    assert normalise_names(_FakeModel(None)) == {}


def test_duplicate_names_resolve_to_the_lowest_id():
    """Deterministic, so a reordered class list can't shift results silently."""
    assert resolve_class_ids({5: "person", 2: "person"}, ["person"]) == [2]
