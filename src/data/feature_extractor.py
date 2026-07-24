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

from src.features.gaze_estimation import estimate_gaze, gaze_ratio
from src.features.eye_analysis import calculate_ear
from src.features.head_pose import calculate_head_pose


# Eye-landmark indices for EAR -- must match src/vision/blink_detection.py.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20      # below this the eye is treated as closed
BLINK_MIN_FRAMES = 2      # eyes closed for >= this many frames counts as one blink

# COCO class ids we care about for proctoring.
YOLO_CLASSES = {"person": 0, "laptop": 63, "cell phone": 67, "book": 73}

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

    def __init__(self, yolo_weights="yolov8n.pt", object_conf=0.5):
        self.object_conf = object_conf

        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )
        self._yolo = YOLO(yolo_weights)

        # Blink state.
        self._closed_frames = 0
        self.blink_total = 0

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
            classes=list(YOLO_CLASSES.values()),
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
