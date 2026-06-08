import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import urllib.request
import os
import csv
from datetime import datetime

# ─── Téléchargement du modèle ───────────────────────────────────────────────
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modèle téléchargé.")

# ─── Initialisation du détecteur ────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.PoseLandmarker.create_from_options(options)

# ─── Initialisation du log CSV ──────────────────────────────────────────────
LOG_FILE = "fall_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event"])

def log_fall():
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "FALL DETECTED"])

# ─── Calcul de l'angle du torse ─────────────────────────────────────────────
def compute_torso_angle(landmarks):
    # Milieu des épaules
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    # Milieu des hanches
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2

    # Vecteur torse
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    # Angle par rapport à la verticale (en degrés)
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    return angle

# ─── Variables de détection ─────────────────────────────────────────────────
FALL_THRESHOLD_ANGLE = 60      # degrés — au-delà = chute possible
CONFIRMATION_FRAMES  = 15      # frames consécutives pour confirmer
fall_counter         = 0
fall_logged          = False

# ─── Webcam ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("Démarrage... Appuie sur Q pour quitter.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = detector.detect(mp_image)

    status = "No person detected"
    color  = (128, 128, 128)
    angle  = None

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]

        # Calcul de l'angle du torse
        angle = compute_torso_angle(landmarks)

        if angle > FALL_THRESHOLD_ANGLE:
            fall_counter += 1
        else:
            fall_counter = 0
            fall_logged  = False

        # Confirmation après N frames consécutives
        if fall_counter >= CONFIRMATION_FRAMES:
            status = "FALL DETECTED"
            color  = (0, 0, 255)
            if not fall_logged:
                log_fall()
                fall_logged = True
        else:
            status = "Standing"
            color  = (0, 255, 0)

        # Dessin des keypoints
        h, w, _ = frame.shape
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

    # ─── Affichage ──────────────────────────────────────────────────────────
    cv2.putText(frame, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    if angle is not None:
        cv2.putText(frame, f"Torso angle: {angle:.1f} deg", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Fall frames: {fall_counter}/{CONFIRMATION_FRAMES}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Fall Detection - Mifolo YEO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Session terminée. Log sauvegardé dans {LOG_FILE}")