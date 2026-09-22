import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# MODEL
# -----------------------------

MODEL_PATH = "hand_landmarker.task"


# -----------------------------
# CREATE HAND LANDMARKER
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# CALLBACK
# -----------------------------

latest_result = None


def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


options.result_callback = result_callback


# -----------------------------
# START MEDIAPIPE
# -----------------------------

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# -----------------------------
# OPEN WEBCAM
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    landmarker.close()
    exit()


print("Webcam started.")
print("Press Q to quit.")


# -----------------------------
# CAMERA LOOP
# -----------------------------

timestamp_ms = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Send frame to MediaPipe
    landmarker.detect_async(
        mp_image,
        timestamp_ms
    )

    timestamp_ms += 1


    # -----------------------------
    # DRAW HAND
    # -----------------------------

    if latest_result is not None:

        for hand in latest_result.hand_landmarks:

            # Draw points
            for landmark in hand:

                x = int(
                    landmark.x * frame.shape[1]
                )

                y = int(
                    landmark.y * frame.shape[0]
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # Hand connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16),
                (13, 17), (17, 18), (18, 19), (19, 20),
                (0, 17)
            ]

            for start, end in connections:

                x1 = int(
                    hand[start].x * frame.shape[1]
                )

                y1 = int(
                    hand[start].y * frame.shape[0]
                )

                x2 = int(
                    hand[end].x * frame.shape[1]
                )

                y2 = int(
                    hand[end].y * frame.shape[0]
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


    # -----------------------------
    # SHOW CAMERA
    # -----------------------------

    cv2.imshow(
        "Hand Tracking",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# CLEANUP
# -----------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Camera closed.")