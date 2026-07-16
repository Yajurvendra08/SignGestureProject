"""
recognize.py
------------
Step 3 of the pipeline: run real-time gesture recognition using the webcam
and the model trained by train_model.py.

Usage:
    python src/recognize.py
    python src/recognize.py --model models/gesture_model.pkl --camera 1

Press 'q' to quit.
"""

import argparse
import os
import sys

import cv2
import joblib

# Must match capture_images.py / train_model.py.
ROI_TOP, ROI_BOTTOM = 100, 400
ROI_LEFT, ROI_RIGHT = 100, 400
IMG_SIZE = 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time gesture recognition.")
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
            "Run capture_images.py and then train_model.py first."
        )
    model = joblib.load(args.model)
    has_confidence = hasattr(model, "predict_proba")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"[ERROR] Could not open webcam at index {args.camera}. "
            "Try a different --camera index, and make sure no other app is using the camera."
        )

    print("[INFO] Starting real-time gesture recognition. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            features = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)

            prediction = model.predict(features)[0]
            label = str(prediction)
            if has_confidence:
                confidence = model.predict_proba(features).max()
                label = f"{prediction} ({confidence:.0%})"

            cv2.rectangle(frame, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
            cv2.putText(frame, f"Prediction: {label}", (10, ROI_TOP - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Live Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
