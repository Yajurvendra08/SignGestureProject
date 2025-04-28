import cv2
import joblib
import numpy as np

model = joblib.load("gesture_model.pkl")

cap = cv2.VideoCapture(0)
print("Starting real-time gesture recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (100, 100)).flatten().reshape(1, -1)

    prediction = model.predict(resized)[0]

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
    cv2.putText(frame, f"Prediction: {prediction}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Live Gesture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()