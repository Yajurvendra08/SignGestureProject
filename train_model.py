import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def load_data(data_dir="gesture_data"):
    X, y = [], []
    labels = os.listdir(data_dir)

    for label in labels:
        folder = os.path.join(data_dir, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100)).flatten()
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("Model accuracy:", model.score(X_test, y_test))

joblib.dump(model, "gesture_model.pkl")
print("Model saved to 'gesture_model.pkl'")