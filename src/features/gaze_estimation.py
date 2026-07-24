LEFT_IRIS = [474, 475, 476, 477]
LEFT_EYE_CORNERS = [362, 263]

RIGHT_IRIS = [469, 470, 471, 472]
RIGHT_EYE_CORNERS = [33, 133]


def get_iris_center(iris_points, landmarks, width, height):

    points = []

    for index in iris_points:
        landmark = landmarks[index] 

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)

    return int(center_x), int(center_y)
    
def calculate_gaze_ratio(eye_corners, iris_center, landmarks, width, height):

    left_corner = landmarks[eye_corners[0]]
    right_corner = landmarks[eye_corners[1]]

    left_x = int(left_corner.x * width)
    right_x = int(right_corner.x * width)

    if left_x > right_x:
        left_x, right_x = right_x, left_x

    eye_width = right_x - left_x

    if eye_width == 0:
        return 0.5

    ratio = (iris_center[0] - left_x) / eye_width

    return ratio



def gaze_ratio(landmarks, width, height):
    """Return the raw horizontal gaze ratio for the left eye.

    ~0.0 -> iris at the left corner, ~1.0 -> at the right corner. This continuous
    value is more useful for a model than the LEFT/RIGHT/CENTER label, which
    throws information away.
    """
    left_iris = get_iris_center(
        LEFT_IRIS,
        landmarks,
        width,
        height
    )

    return calculate_gaze_ratio(
        LEFT_EYE_CORNERS,
        left_iris,
        landmarks,
        width,
        height
    )


def estimate_gaze(landmarks, width, height):
    ratio = gaze_ratio(landmarks, width, height)

    if ratio < 0.35:
        return "LEFT"

    elif ratio > 0.65:
        return "RIGHT"

    else:
        return "CENTER"