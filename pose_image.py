import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# === MODEL OLUŞTURMA ===
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    running_mode = vision.RunningMode.IMAGE,
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=5,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)
detector = vision.PoseLandmarker.create_from_options(options)

# === GÖRÜNTÜ YÜKLEME ===
image = mp.Image.create_from_file("image.jpg")
detection_result = detector.detect(image)

# === SEGMENTASYON MASKESİ VE ÇİZİM ===
np_image = image.numpy_view().copy()
h, w, _ = np_image.shape

# Bağlantılar (pose connections) için sabit yapı
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# Kullanıcı seçimi için değişken
show_only_mask = False  # True: Sadece maskeyi göster, False: Arka plan gri + iskelet çizimi

all_keypoints = []
for i, pose_landmarks in enumerate(detection_result.pose_landmarks):
    person_keypoints = []
    # Her kişinin keypointlerini yazdır
    # print(f"Kişi {i+1} keypointleri:")
    for idx, lm in enumerate(pose_landmarks):
        # print(f"  Keypoint {idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, visibility={lm.visibility:.3f}")
        person_keypoints.append((lm.x, lm.y, lm.z, lm.visibility))
    all_keypoints.append(person_keypoints)

    if detection_result.segmentation_masks:
        mask = detection_result.segmentation_masks[i].numpy_view()
        mask = cv2.resize(mask, (w, h))
        mask_3ch = np.stack((mask,) * 3, axis=-1)
        if show_only_mask:
            # Sadece maskeyi göster
            visualized_mask = (mask_3ch * 255).astype(np.uint8)
            cv2.imshow(f"Segmentation Mask {i+1}", visualized_mask)
        else:
            # Arka plan gri + iskelet çizimi
            bg_image = np.full(np_image.shape, 128, dtype=np.uint8)
            np_image = np.where(mask_3ch > 0.5, np_image, bg_image)

    if not show_only_mask:
        # Landmark noktalarını çiz
        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(np_image, (cx, cy), 3, (0, 255, 0), -1)
        # Bağlantıları çiz
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = pose_landmarks[start_idx]
            end = pose_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(np_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

# === GÖSTER ===
if show_only_mask:
    cv2.imshow("Pose Estimation (Tasks API)", cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()


# ****************************************
# POSE ESTIMATION KEYPOINTLERİNİ YAZDIRMA
print("**********Tüm kişilerin keypointleri**********")
for idx, person in enumerate(all_keypoints):
    print(f"Kişi {idx+1}:")
    for kp_idx, kp in enumerate(person):
        x, y, z, vis = kp
        print(f"  Keypoint {kp_idx}: x={x:.3f}, y={y:.3f}, z={z:.3f}, visibility={vis:.3f}")


# all_keypoints listesinden iskelet çizimi
np_image_copy = np_image.copy()
for person in all_keypoints:

    # Noktaları çiz
    for kp in person:
        x, y = int(kp[0] * w), int(kp[1] * h)
        cv2.circle(np_image_copy, (x, y), 3, (255, 0, 0), -1)

    # Bağlantıları çiz
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(person) and end_idx < len(person):
            x1, y1 = int(person[start_idx][0] * w), int(person[start_idx][1] * h)
            x2, y2 = int(person[end_idx][0] * w), int(person[end_idx][1] * h)
            cv2.line(np_image_copy, (x1, y1), (x2, y2), (255, 0, 255), 2)



cv2.imshow("all_keypoints ile iskelet çizimi", cv2.cvtColor(np_image_copy, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# print("Versiyon: ",mp.__version__) --> 10.21

print()
print("**********İskelet bağlantıları (POSE_CONNECTIONS)**********")
for idx, (start_idx, end_idx) in enumerate(POSE_CONNECTIONS):
    print(f"Bağlantı {idx+1}: Keypoint {start_idx} <-> Keypoint {end_idx}")
