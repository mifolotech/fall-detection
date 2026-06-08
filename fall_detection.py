import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os

# Téléchargement automatique du modèle si absent
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modèle téléchargé.")

# Configuration du détecteur de pose
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)

# Ouverture de la webcam
cap = cv2.VideoCapture(0)
print("Démarrage... Appuie sur Q pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Détection
    result = detector.detect(mp_image)

    status = "No person detected"
    color = (128, 128, 128)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # Y des épaules et chevilles (coordonnées normalisées 0.0 à 1.0)
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        ankle_y    = (landmarks[27].y + landmarks[28].y) / 2

        vertical_diff = ankle_y - shoulder_y

        if vertical_diff < 0.15:
            status = "FALL DETECTED"
            color = (0, 0, 255)
        else:
            status = "Standing"
            color = (0, 255, 0)

        # Dessin manuel des points du squelette
        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.imshow("Fall Detection - Mifolo YEO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()