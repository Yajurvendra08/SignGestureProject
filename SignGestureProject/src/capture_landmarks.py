"""
capture_landmarks.py
---------------------
Recommended Step 1: collect hand-landmark samples for a single gesture,
using Google's pretrained hand detector instead of raw images. Because the
detector already normalizes for hand shape, you need far fewer samples
than the pixel-based pipeline (~50-80 instead of ~150), and it works
reliably across different people, skin tones, lighting, and backgrounds.

Usage:
    python src/capture_landmarks.py --label hello
    python src/capture_landmarks.py --label thanks --num-samples 80 --camera 1

Controls:
    ESC  -> stop early
Runs in AUTO mode by default (captures continuously on a timer). Pass
--manual to press SPACE for each sample instead.
"""

import argparse
import csv
import os
import sys
import time

import cv2

from landmark_utils import create_landmarker, normalize_landmarks, draw_landmarks, to_mp_image, FEATURE_COLUMNS
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "gesture_data", "landmarks.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture hand-landmark samples for one gesture label.")
    parser.add_argument("--label", type=str, default=None,
                         help="Gesture name, e.g. 'hello'. If omitted, you'll be prompted.")
    parser.add_argument("--num-samples", type=int, default=60,
                         help="Target number of samples to capture (default: 60).")
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH,
                         help="CSV file where landmark samples are stored.")
    parser.add_argument("--camera", type=int, default=0,
                         help="Webcam index. Try 1 or 2 if 0 doesn't work (default: 0).")
    parser.add_argument("--manual", action="store_true",
                         help="Press SPACE to save each sample instead of auto-capturing.")
    parser.add_argument("--interval", type=float, default=0.1,
                         help="Seconds between automatic captures (default: 0.1).")
    return parser.parse_args()


def main():
    args = parse_args()

    label = args.label or input("Enter the gesture label (e.g., hello, thanks): ").strip()
    if not label:
        sys.exit("[ERROR] Label cannot be empty.")

    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    file_exists = os.path.isfile(args.csv_path)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"[ERROR] Could not open webcam at index {args.camera}. "
            "Try a different --camera index, and make sure no other app is using the camera."
        )

    landmarker = create_landmarker(running_mode=vision.RunningMode.VIDEO)

    mode_hint = "AUTO capturing... ESC=stop" if not args.manual else "[SPACE=save, ESC=quit]"
    print(f"[INFO] Saving samples to: {args.csv_path}")
    print(f"[INFO] Target: {args.num_samples} samples for '{label}'.")
    print(f"[INFO] {mode_hint}. Show your hand and move it slightly between captures.")

    saved = 0
    last_capture_time = 0.0
    start_time = time.time()

    with open(args.csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + FEATURE_COLUMNS)

        try:
            while saved < args.num_samples:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Failed to read from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                timestamp_ms = int((time.time() - start_time) * 1000)
                result = landmarker.detect_for_video(to_mp_image(frame), timestamp_ms)

                hand_found = bool(result.hand_landmarks)
                if hand_found:
                    hand = result.hand_landmarks[0]
                    draw_landmarks(frame, hand, w, h)

                status = f"{label}: {saved}/{args.num_samples}  {mode_hint}"
                if not hand_found:
                    status += "  (no hand detected)"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Capture Gesture (Landmarks)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

                should_capture = False
                if hand_found:
                    if args.manual:
                        should_capture = key == 32  # SPACE
                    else:
                        now = time.time()
                        if now - last_capture_time >= args.interval:
                            should_capture = True
                            last_capture_time = now

                if should_capture:
                    features = normalize_landmarks(result.hand_landmarks[0])
                    writer.writerow([label] + features.tolist())
                    saved += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()

    print(f"[DONE] {saved} samples saved for '{label}'.")


if __name__ == "__main__":
    main()
