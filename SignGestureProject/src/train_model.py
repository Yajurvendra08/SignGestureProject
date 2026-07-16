"""
train_model.py
---------------
Step 2 of the pipeline: train a classifier on the images collected by
capture_images.py, and save it for recognize.py to use.

Usage:
    python src/train_model.py
    python src/train_model.py --data-dir gesture_data --model-out models/gesture_model.pkl
"""

import argparse
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Must match capture_images.py / recognize.py.
IMG_SIZE = 100

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "gesture_data")
DEFAULT_MODEL_OUT = os.path.join(PROJECT_ROOT, "models", "gesture_model.pkl")

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a gesture recognition model.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                         help="Folder containing one subfolder of images per gesture label.")
    parser.add_argument("--model-out", type=str, default=DEFAULT_MODEL_OUT,
                         help="Where to save the trained model (.pkl).")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of images held out for evaluation (default: 0.2).")
    return parser.parse_args()


def load_data(data_dir):
    X, y = [], []

    if not os.path.isdir(data_dir):
        sys.exit(f"[ERROR] Data folder not found: {data_dir}\nRun capture_images.py first.")

    # Skip hidden/system folders (e.g. .DS_Store on macOS) so this works
    # the same way on Windows, macOS and Linux.
    labels = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    )

    if len(labels) < 2:
        sys.exit(
            f"[ERROR] Found {len(labels)} gesture folder(s) in {data_dir}. "
            "Capture at least 2 different gestures before training."
        )

    for label in labels:
        folder = os.path.join(data_dir, label)
        count = 0
        for img_name in sorted(os.listdir(folder)):
            if not img_name.lower().endswith(VALID_EXTENSIONS):
                continue
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Skipping unreadable image: {img_path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).flatten()
            X.append(img)
            y.append(label)
            count += 1
        print(f"[INFO] {label}: {count} images")

    return np.array(X), np.array(y)


def main():
    args = parse_args()

    print(f"[INFO] Loading images from {args.data_dir} ...")
    X, y = load_data(args.data_dir)
    print(f"[INFO] Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # probability=True lets recognize.py show a confidence score.
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"[INFO] Model accuracy on held-out test set: {accuracy:.2%}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"[DONE] Model saved to '{args.model_out}'")


if __name__ == "__main__":
    main()
