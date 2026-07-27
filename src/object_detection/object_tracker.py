"""Object detection for the live proctoring loop.

Uses the Phase 2 fine-tuned detector when it's present, falling back to stock
COCO weights otherwise. The two models number their classes completely
differently -- 'cell phone' is class 67 of eighty in COCO and class 1 of six in
the fine-tune -- so nothing here may hard-code an id. Classes are resolved by
NAME at load time, which is the only thing the two models agree on.

That mattered: asking a six-class model for class 67 matches nothing, and would
have reported "no phone ever" as though the detector had failed.
"""

import csv
import os
from datetime import datetime

from ultralytics import YOLO

from src.detection.class_ids import resolve_class_ids
from src.thresholds import detector_weights, phone_conf, using_finetuned

LOG_FILE = os.path.join("logs", "detections.csv")

# Objects worth reporting, by name. Anything the loaded model doesn't know is
# simply skipped -- the COCO baseline has no 'headphone', for instance.
INTERESTING = ["cell phone", "book", "laptop", "headphone", "tv", "person"]

# Which of those are suspicious enough to log as an event.
SUSPICIOUS = {"cell phone", "book", "headphone"}


def _ensure_log():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as handle:
            csv.writer(handle).writerow(["timestamp", "object", "confidence"])


_ensure_log()

model = YOLO(detector_weights())

# Resolve names -> ids once, for whichever model actually loaded.
CLASS_IDS = resolve_class_ids(model, INTERESTING)

# This module backs the LIVE app, so it takes the stricter operating point --
# background clutter in an unfamiliar room reads as `cell phone` around 0.35-0.53.
CONF_THRESHOLD = phone_conf(live=True)

_WHICH = (
    "fine-tuned"
    if using_finetuned()
    else "COCO baseline -- phone recall is poor until the Phase 2 fine-tune is in place"
)
print(f"[detector] {detector_weights()} ({_WHICH})")
print(f"[detector] tracking {[model.names[i] for i in CLASS_IDS]} "
      f"at conf >= {CONF_THRESHOLD}")


def detect_objects(frame):
    """Run detection on one frame.

    Returns (raw_results, [{'label': str, 'confidence': float}]).
    """
    results = model(frame, classes=CLASS_IDS, verbose=False)

    detected = []
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if confidence < CONF_THRESHOLD:
            continue

        label = model.names[int(box.cls[0])]
        detected.append({"label": label, "confidence": confidence})

        if label in SUSPICIOUS:
            with open(LOG_FILE, "a", newline="") as handle:
                csv.writer(handle).writerow(
                    [datetime.now(), label, round(confidence, 2)]
                )

    return results, detected
