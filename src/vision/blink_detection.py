import cv2
import mediapipe as mp

from src.features.eye_analysis import calculate_ear

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

LEFT_EYE = [
    33,   # left corner
    160,  # top-left
    158,  # top-right
    133,  # right corner
    153,  # bottom-right
    144   # bottom-left
]

RIGHT_EYE = [
    362,  # right corner
    385,  # top-left
    387,  # top-right
    263,  # left corner
    373,  # bottom-right
    380   # bottom-left
]

from src.thresholds import EAR_CLOSED

# Single source of truth -- this used to be its own literal 0.20, which is how
# it and feature_extractor.py could have drifted apart unnoticed.
EAR_THRESHOLD = EAR_CLOSED

blink_counter = 0
blink_total = 0

cap = cv2.VideoCapture(0)


while True:

    success, frame = cap.read()

    if not success:
        break


    # Flip image like a mirror
    frame = cv2.flip(frame, 1)


    # Get image dimensions
    h, w, _ = frame.shape


    # Convert BGR → RGB for MediaPipe
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Process face landmarks
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        landmarks = face_landmarks.landmark


        left_ear = calculate_ear(
            LEFT_EYE,
            landmarks,
            w,
            h
        )


        right_ear = calculate_ear(
            RIGHT_EYE,
            landmarks,
            w,
            h
        )


        ear = (left_ear + right_ear) / 2

        if ear < EAR_THRESHOLD:

            blink_counter += 1

        else:

            if blink_counter >= 2:
                blink_total += 1

            blink_counter = 0    
    
        cv2.putText(
        frame,
        f"EAR: {ear:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
        cv2.putText(
        frame,
        f"Blinks: {blink_total}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )
    cv2.imshow(
        "Blink Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()