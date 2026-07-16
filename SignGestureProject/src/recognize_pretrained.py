"""
recognize_pretrained.py
-------------------------
Zero-training gesture recognition. Uses Google's pretrained MediaPipe
Gesture Recognizer, trained on real hand gestures from many different
people, so it works immediately -- no capturing images, no training step.

It recognizes 7 common hand gestures out of the box:
    Open Palm / Hello, Closed Fist, Thumbs Up / Yes, Thumbs Down / No,
    Victory / Peace, Pointing Up, I Love You (ASL)

This is NOT full sign-language recognition (ASL/BSL/etc. have thousands of
signs, many involving motion over time) -- it's a solid, ready-to-use base
for the common gestures people actually use day to day. For anything
beyond these 7, use the landmark pipeline (capture_landmarks.py +
train_landmark_model.py) to teach the system new signs; if you've trained
a custom model, this script will automatically fall back to it whenever
the pretrained model doesn't recognize a confident built-in gesture.

Usage:
    python src/recognize_pretrained.py
    python src/recognize_pretrained.py --camera 1

Press 'q' to quit.
"""

import argparse
import os
import time

import cv2

from landmark_utils import (
    create_gesture_recognizer,
    draw_landmarks,
    to_mp_image,
    normalize_landmarks,
    PRETRAINED_GESTURE_LABELS,
)
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CUSTOM_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_landmark_model.pkl")

CONFIDENCE_THRESHOLD = 0.6


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-training real-time gesture recognition.")
    parser.add_argument("--camera", type=int, default=0,
                         help="Webcam index. Try 1 or 2 if 0 doesn't work (default: 0).")
    return parser.parse_args()


def load_custom_model():
    if not os.path.isfile(CUSTOM_MODEL_PATH):
        return None
    import joblib
    print(f"[INFO] Found custom-trained model too: {CUSTOM_MODEL_PATH} (used as fallback)")
    return joblib.load(CUSTOM_MODEL_PATH)


def main():
    args = parse_args()
    custom_model = load_custom_model()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            f"[ERROR] Could not open webcam at index {args.camera}. "
            "Try a different --camera index, and make sure no other app is using the camera."
        )

    recognizer = create_gesture_recognizer(running_mode=vision.RunningMode.VIDEO)
    start_time = time.time()

    print("[INFO] Starting zero-training gesture recognition. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = recognizer.recognize_for_video(to_mp_image(frame), timestamp_ms)

            label_text = "Show your hand to the camera"
            color = (0, 0, 255)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_landmarks(frame, hand, w, h)

                top_gesture = None
                if result.gestures:
                    top_gesture = max(result.gestures[0], key=lambda c: c.score)

                if top_gesture and top_gesture.score >= CONFIDENCE_THRESHOLD and top_gesture.category_name != "None":
                    friendly = PRETRAINED_GESTURE_LABELS.get(top_gesture.category_name, top_gesture.category_name)
                    label_text = f"{friendly} ({top_gesture.score:.0%})"
                    color = (0, 255, 0)
                elif custom_model is not None:
                    features = normalize_landmarks(hand).reshape(1, -1)
                    prediction = custom_model.predict(features)[0]
                    conf_text = ""
                    if hasattr(custom_model, "predict_proba"):
                        conf_text = f" ({custom_model.predict_proba(features).max():.0%})"
                    label_text = f"{prediction}{conf_text}  [custom]"
                    color = (255, 200, 0)
                else:
                    label_text = "Gesture not recognized"

            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Live Gesture Recognition (Pretrained)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()


if __name__ == "__main__":
    main()
