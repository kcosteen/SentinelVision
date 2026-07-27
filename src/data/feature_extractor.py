"""
Per-frame feature extraction for building a behavior-model training set (Phase 1).

This runs the vision models ONCE per frame and turns each frame into a flat row of
numeric features -- the same signals the live app reacts to (gaze, head pose,
blink / EAR, face count, objects). Logging these to CSV while you record is what
turns raw webcam sessions into a labeled dataset you can train on.

Design note: model *inference* lives here, but the per-signal math is reused from
the existing pure functions (`gaze_ratio` / `estimate_gaze`, `calculate_ear`,
`calculate_head_pose`). That keeps a single source of truth for each feature, so
the numbers logged for training match what the live pipeline computes.
"""

import cv2
import mediapipe as mp
from ultralytics import YOLO

from src.detection.class_ids import resolve_class_ids
from src.features.gaze_estimation import estimate_gaze, gaze_ratio
from src.features.eye_analysis import calculate_ear
from src.features.head_pose import calculate_head_pose
from src.thresholds import (
    BASELINE_WEIGHTS,
    BLINK_MIN_FRAMES,
    EAR_CLOSED,
    FINETUNED_WEIGHTS,
    detector_weights,
    phone_conf,
)


# Eye-landmark indices for EAR, as used by src/calibration/calibrate_ear.py
# when it swept the threshold -- the two must agree or the number doesn't apply.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Both now come from src/thresholds.py, which records whether each was measured
# or guessed. EAR_CLOSED is calibrated (Cohen's d 4.11 between open and closed).
EAR_THRESHOLD = EAR_CLOSED

# Objects worth logging, BY NAME -- never by id. See src/detection/class_ids.py:
# 'cell phone' is 67 in COCO and 1 in the proctoring fine-tune, so the previous
# hard-coded {"person": 0, "laptop": 63, "cell phone": 67, "book": 73} silently
# resolved to nothing but 'book' against the fine-tune, logging person_count 0
# and phone_conf 0.0 on every frame.
OBJECT_NAMES = ["person", "laptop", "cell phone", "book"]

# The measurement columns this extractor produces (order defines the CSV order).
MEASUREMENT_FIELDS = [
    "face_count",
    "gaze_ratio",
    "gaze_direction",
    "ear",
    "eyes_closed",
    "blink_total",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "person_count",
    "phone_detected",
    "phone_conf",
    "book_detected",
]


def _round(value, ndigits=4):
    """Round for readable CSVs; None -> "" (empty cell)."""
    return round(value, ndigits) if value is not None else ""


class FeatureExtractor:
    """Turns a video frame into a dict of behavior features.

    The only state carried across frames is the blink counter (a blink spans
    several frames), so create ONE extractor per recording session.
    """

    def __init__(self, yolo_weights=None, object_conf=None):
        # Default to the best available detector and ITS calibrated threshold,
        # rather than the old hard-coded ('yolov8n.pt', 0.5). That pairing was
        # measured at recall 0.064 on real proctoring frames -- it logged a
        # phone in ~6% of the frames that had one, which is why the Phase 1
        # phone model learned head-down posture as a proxy instead.
        self._weights = yolo_weights or detector_weights()
        self._yolo = YOLO(self._weights)
        self.object_conf = object_conf if object_conf is not None else phone_conf()

        # Resolve names -> ids once, for whichever model actually loaded.
        self.class_ids = resolve_class_ids(self._yolo, OBJECT_NAMES)
        if not self.class_ids:
            raise SystemExit(
                f"{self._weights} knows none of {OBJECT_NAMES}: "
                f"{sorted((self._yolo.names or {}).values())}"
            )

        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )

        # Blink state.
        self._closed_frames = 0
        self.blink_total = 0

    def describe(self):
        """One line naming the detector behind these features.

        Worth printing wherever features are logged: rows produced by the COCO
        baseline and rows produced by the fine-tune are NOT comparable, and a
        dataset silently mixing the two would train on a moving target.
        """
        # Keyed off the weights actually loaded, not off which files exist on
        # disk -- an explicitly-passed path must not be described as the default.
        if self._weights == FINETUNED_WEIGHTS:
            which = "Phase 2 fine-tune"
        elif self._weights == BASELINE_WEIGHTS:
            which = "stock COCO baseline -- weak on phones (recall 0.064 at conf 0.5)"
        else:
            which = f"{len(self._yolo.names or {})} classes"
        tracking = [self._yolo.names[i] for i in self.class_ids]
        return (f"{self._weights} ({which}), conf >= {self.object_conf}, "
                f"tracking {tracking}")

    def extract(self, frame):
        """Return a dict of MEASUREMENT_FIELDS for one frame."""
        # Mirror the frame so left/right match the on-screen view -- consistent
        # with the live gaze / blink / head-pose modules, which all flip.
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Start with empty cells; face-mesh features stay empty if no face is found.
        row = {field: "" for field in MEASUREMENT_FIELDS}

        # --- Face count (how many people are present) ---
        detections = self._face_detection.process(rgb).detections
        row["face_count"] = len(detections) if detections else 0

        # --- Face-mesh features: gaze, EAR/blink, head pose ---
        mesh = self._face_mesh.process(rgb)
        if mesh.multi_face_landmarks:
            landmarks = mesh.multi_face_landmarks[0].landmark

            row["gaze_ratio"] = _round(gaze_ratio(landmarks, width, height))
            row["gaze_direction"] = estimate_gaze(landmarks, width, height)

            left_ear = calculate_ear(LEFT_EYE, landmarks, width, height)
            right_ear = calculate_ear(RIGHT_EYE, landmarks, width, height)
            ear = (left_ear + right_ear) / 2
            row["ear"] = _round(ear)

            eyes_closed = ear < EAR_THRESHOLD
            row["eyes_closed"] = int(eyes_closed)
            self._update_blinks(eyes_closed)
            row["blink_total"] = self.blink_total

            pitch, yaw, roll = calculate_head_pose(landmarks, width, height)
            row["head_pitch"] = _round(pitch)
            row["head_yaw"] = _round(yaw)
            row["head_roll"] = _round(roll)

        # --- Objects: phone / book / people ---
        self._extract_objects(frame, row)

        return row

    def _update_blinks(self, eyes_closed):
        """A blink = eyes closed for >= BLINK_MIN_FRAMES consecutive frames."""
        if eyes_closed:
            self._closed_frames += 1
        else:
            if self._closed_frames >= BLINK_MIN_FRAMES:
                self.blink_total += 1
            self._closed_frames = 0

    def _extract_objects(self, frame, row):
        results = self._yolo(
            frame,
            classes=self.class_ids,
            verbose=False,
        )
        names = self._yolo.names

        person_count = 0
        phone_conf = 0.0
        book_conf = 0.0
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.object_conf:
                continue
            label = names[int(box.cls[0])]
            if label == "person":
                person_count += 1
            elif label == "cell phone":
                phone_conf = max(phone_conf, conf)
            elif label == "book":
                book_conf = max(book_conf, conf)

        row["person_count"] = person_count
        row["phone_detected"] = int(phone_conf > 0)
        row["phone_conf"] = _round(phone_conf)
        row["book_detected"] = int(book_conf > 0)
