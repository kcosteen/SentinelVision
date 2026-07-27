"""Resolving class ids from class NAMES, for whichever detector is loaded.

Every model in this project numbers its classes differently. 'cell phone' is
class 67 of eighty in COCO, class 1 of six in the Roboflow proctoring fine-tune,
and class 0 of one in the single-class set `prepare_dataset.py` builds. The name
is the only thing all three agree on.

Hard-coding an id is uniquely nasty here because the failure is **silent and
plausible**: asking a six-class model for class 67 matches nothing, so the
pipeline reports "no phone in any frame" and looks like a detector that failed
rather than a lookup that was wrong. That mistake has been fixed one file at a
time, each with its own copy of the same name set -- which is how the copies
drifted. This module is the one place that knowledge lives.

Deliberately dependency-free (no ultralytics, no cv2) so any module can import it.
"""

# Every name a phone has gone by across the datasets surveyed in docs/DATASETS.md.
# Compared case-insensitively after stripping, so 'Cell Phone' matches.
PHONE_CLASS_NAMES = {"cell phone", "phone", "mobile phone", "mobile_phone"}


def normalise_names(source):
    """Coerce a model's class names into {id: lowercase name}.

    Accepts what the various sources actually hand us: an ultralytics model, its
    `.names` dict ({0: 'book', ...}), or a plain list from a data.yaml `names:`
    key (where position is the id).
    """
    names = getattr(source, "names", source)
    if names is None:
        return {}
    if isinstance(names, dict):
        pairs = names.items()
    else:
        pairs = enumerate(names)
    return {int(index): str(name).strip().lower() for index, name in pairs}


def phone_class_index(source, required=True):
    """The id of the phone-like class, looked up by name.

    Returns None when the model has no phone class and `required` is False.
    Raising by default is intentional: a missing phone class means the caller is
    about to measure something other than what it thinks it is.
    """
    names = normalise_names(source)
    for index, name in sorted(names.items()):
        if name in PHONE_CLASS_NAMES:
            return index

    if not required:
        return None
    raise SystemExit(
        "No phone-like class in this model. Looked for "
        f"{sorted(PHONE_CLASS_NAMES)}, found {sorted(names.values())}."
    )


def resolve_class_ids(source, wanted):
    """Ids for each name in `wanted` that the model actually knows.

    Unknown names are skipped rather than raising -- the COCO baseline has no
    'headphone' class, and that's a fact about the model, not an error. Returns
    ids in `wanted` order so callers can log what they asked for versus got.
    """
    names = normalise_names(source)
    by_name = {}
    for index, name in sorted(names.items()):
        by_name.setdefault(name, index)

    return [by_name[w.strip().lower()] for w in wanted if w.strip().lower() in by_name]
