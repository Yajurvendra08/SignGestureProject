"""
recognize_landmarks.py
------------------------
Recommended Step 3: real-time gesture recognition using hand landmarks.

Usage:
    python src/recognize_landmarks.py
    python src/recognize_landmarks.py --model models/gesture_landmark_model.pkl --camera 1

Press 'q' to quit.
"""

import argparse
import os
import sys
import time

import cv2
import joblib

from landmark_utils import create_landmarker, normalize_landmarks, draw_landmarks, to_mp_image
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_landmark_model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time gesture recognition using hand landmarks.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                         help="Path to the trained .pkl model.")
    parser.add_argument("--camera", type=int, default=0,
                         help="Webcam index. Try 1 or 2 if 0 doesn't work (default: 0).")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        sys.exit(
            f"[ERROR] Model file not found: {args.model}\n"
            "Run capture_landmarks.py and then train_landmark_model.py first."
        )
    model = joblib.load(args.model)
    has_confidence = hasattr(model, "predict_proba")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"[ERROR] Could not open webcam at index {args.camera}. "
            "Try a different --camera index, and make sure no other app is using the camera."
        )

    landmarker = create_landmarker(running_mode=vision.RunningMode.VIDEO)
    start_time = time.time()

    print("[INFO] Starting real-time gesture recognition. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = landmarker.detect_for_video(to_mp_image(frame), timestamp_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_landmarks(frame, hand, w, h)

                features = normalize_landmarks(hand).reshape(1, -1)
                prediction = model.predict(features)[0]
                label = str(prediction)
                if has_confidence:
                    confidence = model.predict_proba(features).max()
                    label = f"{prediction} ({confidence:.0%})"

                cv2.putText(frame, f"Prediction: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show your hand to the camera", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Live Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
