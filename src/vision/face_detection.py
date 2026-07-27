"""Face presence and count.

Measures only -- it does not draw. This used to call
`mp_drawing.draw_detection(frame, ...)`, painting boxes and keypoints onto the
caller's array IN PLACE, and main.py then handed that same array to YOLO. The
detector was reading graphics drawn over the subject: on a real clip those ~3,000
altered pixels dropped `person` from 0.63 to no detection at all.

All drawing now lives in src/vision/hud.py, which only ever receives the display
copy. Keeping measurement and rendering apart is what makes that bug impossible
rather than merely fixed.
"""

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


def detect_faces(frame):
    """Returns (raw results, face count). The frame is never modified."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_detection.process(rgb_frame)

    face_count = len(face_results.detections) if face_results.detections else 0

    return face_results, face_count