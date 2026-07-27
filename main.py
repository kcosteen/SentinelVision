import cv2

from src.vision.face_detection import detect_faces
from src.vision.gaze_detection import detect_gaze
from src.features.head_pose import calculate_head_pose
from src.object_detection.object_tracker import detect_objects
from src.utils.event_logger import log_event

from src.behavior.proctor_analyzer import ProctorAnalyzer

def main():

    # Initialize analyzer
    analyzer = ProctorAnalyzer()

    # Open webcam
    camera = cv2.VideoCapture(0)

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        # Keep a pristine copy for anything that MEASURES.
        #
        # detect_faces draws its boxes and keypoints onto the frame it is given,
        # in place. Passing that same frame on to YOLO means the detector sees
        # graphics painted over the person: measured on a real clip, those ~3k
        # altered pixels dropped `person` from 0.63 to no detection at all.
        # So: `frame` is what the user sees and may be drawn on, `source` is what
        # the models are allowed to measure.
        source = frame.copy()

        # ------------------------
        # Face Detection  (draws on `frame`)
        # ------------------------
        face_results, face_count = detect_faces(frame)

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
        yolo_results, detected_objects = detect_objects(source)

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
        # Display information
        # ------------------------
        # Drawn on `frame` (the display copy), never on `source`.
        yaw_text = f"{abs(head_yaw):.0f} deg" if head_yaw is not None else "n/a"

        cv2.putText(
            frame,
            f"Gaze: {gaze}   Head: {yaw_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Score: {analyzer.score}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        status = "Calibrating scene..." if calibrating else analyzer.get_status()

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        y = 130
        for event in ([] if calibrating else events):
            cv2.putText(
                frame,
                event,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            y += 30

            

        cv2.imshow("AI Proctor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()