"""
capture_images.py
------------------
Step 1 of the pipeline: collect webcam images for a single gesture label.

Usage:
    python src/capture_images.py --label hello
    python src/capture_images.py --label thanks --num-images 200 --camera 1

Controls (while the camera window is focused):
    SPACE  -> save the current frame's Region of Interest (ROI)
    ESC    -> stop early and exit

Notes for beginners:
    - Keep your hand inside the green box (the ROI). Everything outside the
      box is ignored, so a messy background elsewhere in the room is fine.
    - Aim for 100-200 images per gesture, captured from slightly different
      angles/distances, for a more reliable model.
    - Images are appended to an existing folder, so you can re-run this
      script later to add more samples for the same label without
      overwriting what you already collected.
"""

import argparse
import os
import sys
import time

import cv2

# --- Constants shared with train_model.py and recognize.py ---------------
# Keep these identical across all three scripts. If you change the ROI
# coordinates or IMG_SIZE here, update the other two files as well, or the
# trained model will not match what recognize.py feeds into it.
ROI_TOP, ROI_BOTTOM = 100, 400
ROI_LEFT, ROI_RIGHT = 100, 400

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "gesture_data")


def parse_args():
    parser = argparse.ArgumentParser(description="Capture webcam images for one gesture label.")
    parser.add_argument("--label", type=str, default=None,
                         help="Gesture name, e.g. 'hello'. If omitted, you'll be prompted.")
    parser.add_argument("--num-images", type=int, default=150,
                         help="Target number of images to capture (default: 150).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                         help="Folder where gesture image folders are stored.")
    parser.add_argument("--camera", type=int, default=0,
                         help="Webcam index. Try 1 or 2 if 0 doesn't work (default: 0).")
    parser.add_argument("--auto", action="store_true",
                         help="Capture automatically on a timer instead of pressing SPACE each time. Much faster.")
    parser.add_argument("--interval", type=float, default=0.15,
                         help="Seconds between automatic captures when --auto is used (default: 0.15).")
    return parser.parse_args()


def main():
    args = parse_args()

    label = args.label or input("Enter the gesture label (e.g., hello, thanks): ").strip()
    if not label:
        sys.exit("[ERROR] Label cannot be empty.")

    save_path = os.path.join(args.data_dir, label)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"[ERROR] Could not open webcam at index {args.camera}. "
            "Try a different --camera index, and make sure no other app is using the camera."
        )

    # Continue numbering so re-running this script doesn't overwrite existing images.
    existing = [f for f in os.listdir(save_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    img_count = len(existing)

    print(f"[INFO] Saving to: {save_path}")
    print(f"[INFO] Starting at image #{img_count}. Target: {args.num_images} images.")
    if args.auto:
        print(f"[INFO] AUTO mode: capturing every {args.interval}s. Move your hand slightly between shots. Press ESC to stop early.")
    else:
        print("[INFO] Press SPACE to save the ROI, ESC to stop early.")

    last_capture_time = 0.0

    try:
        while img_count < args.num_images:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]

            mode_hint = "AUTO capturing... ESC=stop" if args.auto else "[SPACE=save, ESC=quit]"
            display = frame.copy()
            cv2.rectangle(display, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
            cv2.putText(display, f"{label}: {img_count}/{args.num_images}  {mode_hint}",
                        (10, ROI_TOP - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Capture Gesture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            if args.auto:
                now = time.time()
                if now - last_capture_time >= args.interval:
                    img_path = os.path.join(save_path, f"{label}_{img_count}.jpg")
                    cv2.imwrite(img_path, roi)
                    img_count += 1
                    last_capture_time = now
            elif key == 32:  # SPACE (manual mode only)
                img_path = os.path.join(save_path, f"{label}_{img_count}.jpg")
                cv2.imwrite(img_path, roi)
                print(f"[SAVED] {img_path}")
                img_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[DONE] {img_count} images saved for '{label}'.")


if __name__ == "__main__":
    main()
