import cv2
import os

def collect_images(label, save_dir="gesture_data", num_images=100):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_dir = os.path.join(save_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"Collecting images for '{label}'... Press 'q' to quit early.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi = frame[100:400, 100:400]
        cv2.imshow("ROI", roi)

        file_path = os.path.join(label_dir, f"{label}_{count}.jpg")
        cv2.imwrite(file_path, roi)
        count += 1

        cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
        cv2.putText(frame, f"Images: {count}/{num_images}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to {label_dir}")