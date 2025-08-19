import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=5,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.6,
)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture("football.mp4")

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    h, w, _ = frame.shape


    detection_result = detector.detect_for_video(mp_image, int(cap.get(cv2.CAP_PROP_POS_MSEC)))
    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:

            for lm in pose_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start = pose_landmarks[start_idx]
                    end = pose_landmarks[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.imshow("Pose Estimation (Manual)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
