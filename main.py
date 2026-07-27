import time

import cv2

from src.vision import hud
from src.vision.face_detection import detect_faces
from src.vision.gaze_detection import detect_gaze
from src.features.head_pose import calculate_head_pose
from src.object_detection.object_tracker import (
    CONF_THRESHOLD,
    detect_objects,
    using_finetuned,
)
from src.utils.event_logger import log_event

from src.behavior.proctor_analyzer import EVENT_POINTS, ProctorAnalyzer

def main():

    # Initialize analyzer
    analyzer = ProctorAnalyzer()

    # Open webcam
    camera = cv2.VideoCapture(0)

    # Named so the HUD can say which detector produced what's on screen -- the
    # fine-tune and the COCO fallback are not interchangeable.
    detector_label = (
        f"fine-tuned yolov8n  conf {CONF_THRESHOLD}" if using_finetuned()
        else f"COCO yolov8n (fallback)  conf {CONF_THRESHOLD}"
    )

    frames_drawn = 0
    started = time.time()

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        # Keep a pristine copy for anything that MEASURES.
        #
        # `source` is what the models are allowed to see; `frame` is what the
        # viewer sees and the only thing that may be drawn on. detect_faces used
        # to paint boxes onto the array it was handed, and YOLO then read those
        # graphics -- on a real clip, ~3,000 altered pixels dropped `person` from
        # 0.63 to no detection at all. Drawing now lives solely in src/vision/hud.py,
        # which never receives `source`, so that class of bug cannot recur.
        source = frame.copy()

        # ------------------------
        # Face Detection
        # ------------------------
        face_results, face_count = detect_faces(source)

        # ------------------------
        # Gaze Detection
        # ------------------------
        gaze_results, gaze = detect_gaze(source)

        # ------------------------
        # Head Pose
        # ------------------------
        # Reuses the face mesh detect_gaze just ran -- solving PnP is cheap, but
        # a second mesh pass would not be. This is the CALIBRATED looking-away
        # signal (|yaw| >= 30 deg, f1 0.869 on the Gourier ground truth), so it
        # is what the analyzer decides on; gaze is only its fallback.
        head_yaw = None

        if gaze_results.multi_face_landmarks:
            h, w, _ = source.shape

            _, head_yaw, _ = calculate_head_pose(
                gaze_results.multi_face_landmarks[0].landmark,
                w,
                h
            )

        # ------------------------
        # Object Detection
        # ------------------------
        _, detected_objects = detect_objects(source)

        # ------------------------
        # Analyze Behavior
        # ------------------------
        events = analyzer.analyze(
            detected_objects,
            face_count,
            gaze,
            head_yaw
        )

        # Bleed the score down first, so a stale flag stops defining the run.
        analyzer.decay()

        # Watch, but don't judge, until the static filter has learned the scene.
        calibrating = analyzer.is_calibrating()

        if not calibrating:

            # Update suspicion score
            for event in events:

                points = analyzer.add_score(event)

                log_event(
                    event,
                    points,
                    analyzer.score
                )

        # ------------------------
        # Display
        # ------------------------
        # Everything below draws on `frame`, the display copy -- never on
        # `source`, which is what the models measure.
        frames_drawn += 1
        elapsed = time.time() - started
        fps = frames_drawn / elapsed if elapsed > 0 else None

        status = analyzer.get_status()

        hud.draw_faces(frame, face_results)
        hud.draw_detections(frame, detected_objects)
        hud.draw_status(frame, status, analyzer.score, calibrating=calibrating)
        hud.draw_signals(frame, gaze, head_yaw, fps=fps, detector=detector_label)
        hud.draw_events(frame, [] if calibrating else events, weights=EVENT_POINTS)
        hud.draw_title(frame, subtitle="fine-tuned YOLOv8n + calibrated thresholds")

        cv2.imshow("SentinelVision", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()