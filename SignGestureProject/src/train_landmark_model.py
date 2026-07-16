"""
train_landmark_model.py
------------------------
Recommended Step 2: train a classifier on the hand-landmark samples
collected by capture_landmarks.py.

Usage:
    python src/train_landmark_model.py
"""

import argparse
import os
import sys
import csv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "gesture_data", "landmarks.csv")
DEFAULT_MODEL_OUT = os.path.join(PROJECT_ROOT, "models", "gesture_landmark_model.pkl")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a gesture recognition model from hand-landmark samples.")
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH,
                         help="CSV file produced by capture_landmarks.py.")
    parser.add_argument("--model-out", type=str, default=DEFAULT_MODEL_OUT,
                         help="Where to save the trained model (.pkl).")
    parser.add_argument("--test-size", type=float, default=0.2,
                         help="Fraction of samples held out for evaluation (default: 0.2).")
    return parser.parse_args()


def load_data(csv_path):
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] {csv_path} not found.\nRun capture_landmarks.py first.")

    X, y = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            y.append(row[0])
            X.append([float(v) for v in row[1:]])

    return np.array(X), np.array(y)


def main():
    args = parse_args()

    print(f"[INFO] Loading samples from {args.csv_path} ...")
    X, y = load_data(args.csv_path)

    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        sys.exit(f"[ERROR] Found only {len(labels)} gesture(s). Capture at least 2 different gestures.")
    for label, count in zip(labels, counts):
        print(f"[INFO] {label}: {count} samples")
    print(f"[INFO] Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"[INFO] Model accuracy on held-out test set: {accuracy:.2%}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"[DONE] Model saved to '{args.model_out}'")


if __name__ == "__main__":
    main()
