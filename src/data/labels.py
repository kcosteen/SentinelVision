"""
Ground-truth behavior labels for recorded sessions.

Proctoring is a **multi-label** problem: several behaviors can be true at the same
time (e.g. looking away *while* holding a phone). So each behavior gets its own
0/1 column rather than a single mutually-exclusive class. A clip can turn on more
than one.

These labels are the *targets* a model learns to predict — which is different
from the *features* the extractor measures. For example `phone_detected` is a
feature (what YOLO thought it saw); `label_phone` is the label (the human-
confirmed truth for that clip). Keeping them separate is important: you train the
model to predict the label from the features.
"""

BEHAVIOR_LABELS = ["looking_away", "phone", "multiple_people", "absent"]

# Column names written to the CSV, prefixed so labels never collide with feature
# columns (e.g. the label `label_phone` vs. the feature `phone_detected`).
LABEL_COLUMNS = [f"label_{name}" for name in BEHAVIOR_LABELS]


def parse_labels(labels_arg):
    """Turn a CLI string like "looking_away,phone" into {label_x: 0/1, ...}.

    - "normal", "", or None  -> all zeros (no abnormal behavior present).
    - Multiple names          -> each corresponding column set to 1.
    - An unknown name          -> ValueError (so typos fail loudly, not silently).
    """
    columns = {col: 0 for col in LABEL_COLUMNS}

    if not labels_arg:
        return columns

    names = [name.strip() for name in labels_arg.split(",") if name.strip()]
    for name in names:
        if name == "normal":
            continue  # "normal" simply means none of the abnormal labels are on
        if name not in BEHAVIOR_LABELS:
            raise ValueError(
                f"Unknown label '{name}'. "
                f"Valid labels: {BEHAVIOR_LABELS} (or 'normal')."
            )
        columns[f"label_{name}"] = 1

    return columns
