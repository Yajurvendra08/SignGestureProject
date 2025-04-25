import cv2
import os

# Ask the user for the gesture name
gesture_name = input("Enter the gesture label (e.g., hello, thanks): ").strip()

# Folder where images will be saved
save_path = os.path.join("gesture_data", gesture_name)
os.makedirs(save_path, exist_ok=True)

# Start capturing from webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press SPACE to capture an image, ESC to exit.")

img_count = len(os.listdir(save_path))  # Continue numbering if images already exist
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame")
        break

    # Show the frame
    cv2.imshow(f"Gesture: {gesture_name} - SPACE to Save, ESC to Exit", frame)

    key = cv2.waitKey(1)
    
    if key == 27:  # ESC key to quit
        break
    elif key == 32:  # SPACE to save image
        img_name = os.path.join(save_path, f"{gesture_name}_{img_count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"[SAVED] {img_name}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
