"""
landmark_utils.py
------------------
Shared helpers for the hand-landmark pipeline (capture_landmarks.py,
train_landmark_model.py, recognize_landmarks.py).

Why landmarks instead of raw pixels?
Google's pretrained MediaPipe Hand Landmarker finds 21 key points on the
hand (fingertips, knuckles, wrist...) in every frame. Classifying the
*shape* formed by these points -- instead of raw pixel colors -- is far
more robust to different skin tones, lighting, backgrounds, and camera
distances, and needs far less training data (tens of samples per gesture
instead of hundreds of images).
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

NUM_LANDMARKS = 21
FEATURE_COLUMNS = [f"lm{i:02d}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

HAND_LANDMARKER_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Pretrained model that recognizes 7 common hand gestures out of the box --
# no training or data capture required.
GESTURE_RECOGNIZER_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_recognizer.task")
GESTURE_RECOGNIZER_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# MediaPipe's built-in gesture categories, mapped to plain-English labels.
PRETRAINED_GESTURE_LABELS = {
    "None": "No confident gesture",
    "Closed_Fist": "Closed Fist",
    "Open_Palm": "Open Palm / Hello",
    "Pointing_Up": "Pointing Up",
    "Thumb_Down": "Thumbs Down / No",
    "Thumb_Up": "Thumbs Up / Yes",
    "Victory": "Victory / Peace",
    "ILoveYou": "I Love You (ASL)",
}

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def download_model_if_missing(url, path, size_hint="a few MB"):
    """Download a MediaPipe .task model file on first run, if not already present."""
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[INFO] Downloading pretrained model (one-time, ~{size_hint})...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] Saved to {path}")
    except Exception as exc:
        raise SystemExit(
            "[ERROR] Could not download the pretrained model automatically.\n"
            f"Reason: {exc}\n"
            "If you're behind a firewall/proxy, download it manually from:\n"
            f"  {url}\n"
            f"and save it as:\n  {path}"
        )


def ensure_model_downloaded():
    """Kept for backwards compatibility: downloads the hand landmark model."""
    download_model_if_missing(HAND_LANDMARKER_URL, HAND_LANDMARKER_PATH, "8 MB")


def create_landmarker(running_mode=vision.RunningMode.VIDEO, num_hands=1):
    ensure_model_downloaded()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_gesture_recognizer(running_mode=vision.RunningMode.VIDEO, num_hands=1):
    """Zero-training recognizer for 7 common gestures (see PRETRAINED_GESTURE_LABELS)."""
    download_model_if_missing(GESTURE_RECOGNIZER_URL, GESTURE_RECOGNIZER_PATH, "8 MB")
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=GESTURE_RECOGNIZER_PATH),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.GestureRecognizer.create_from_options(options)


def to_mp_image(bgr_frame):
    rgb = bgr_frame[:, :, ::-1]
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def normalize_landmarks(hand_landmarks):
    """
    Convert 21 (x, y, z) points into a translation- and scale-invariant
    feature vector, so the model learns hand *shape*, not where the hand
    is in the frame or how close it is to the camera.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    pts -= pts[0]                          # translate: wrist becomes the origin
    scale = np.linalg.norm(pts[9])         # wrist -> middle-finger knuckle distance
    if scale < 1e-6:
        scale = 1e-6
    pts /= scale                           # scale-invariant
    return pts.flatten()


def draw_landmarks(frame, hand_landmarks, width, height):
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
