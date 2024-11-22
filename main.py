from ultralytics import YOLO
import cv2
import os

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not access the camera.")
    exit()

# Create a folder for the screenshots if it doesn't exist
os.makedirs("screenshots", exist_ok=True)
screenshot_count = 0

# Open a file to save the detections (only opened once)
with open("detections.txt", "w") as f:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing video.")
            break

        # Perform detections
        results = model(frame)

        # Save the detections to the file
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]  # Class name
                confidence = box.conf.item()  # Convert the confidence to a Python number
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert the tensor to a list and then to integers
                f.write(f"{class_name} {confidence:.2f} {x1} {y1} {x2} {y2}\n")

        # Show the frame with the detections
        annotated_frame = results[0].plot()
        cv2.imshow("Real-time detection", annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit with 'q'
            break
        elif key == ord(' '):  # Spacebar to save a screenshot
            screenshot_path = f"screenshots/frame_{screenshot_count}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Screenshot saved at: {screenshot_path}")
            screenshot_count += 1

cap.release()
cv2.destroyAllWindows()
