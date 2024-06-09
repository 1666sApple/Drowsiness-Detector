from ultralytics import YOLO
import cv2
import time
import logging
import pygame  # For non-Windows systems
import os
from datetime import datetime

# Set up logging
logging.basicConfig(filename='drowsiness_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")

# Get the class names from the model
class_names = model.model.names

# Create a folder to save drowsy frames
drowsy_frames_folder = "drowsy_frames"
if not os.path.exists(drowsy_frames_folder):
    os.makedirs(drowsy_frames_folder)

# Function to trigger an alert and save the drowsy frame
def trigger_alert(frame):
    # Code to trigger an alert (e.g., play a sound, send a notification, etc.)
    print("Drowsiness detected for 3 seconds! Alert triggered.")
    logging.info("Drowsiness detected for 3 seconds")

    # Play a sound alert for 5 seconds on repeat
    pygame.mixer.init()
    pygame.mixer.music.load("beep-warning-6387.mp3")
    pygame.mixer.music.play(-1)  # -1 means loop indefinitely

    # Add the timestamp to the frame
    # Get the current timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # Split timestamp into date and time parts
    date_part, time_part = timestamp.split('_')

    # Add the timestamp to the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Date: {date_part}   Time: {time_part}"
    text_size, _ = cv2.getTextSize(text, font, 0.3, 1)
    frame = cv2.putText(frame, text, (10, 30), font, 0.3, (0, 0, 255), 1)

    # Save the frame with the timestamp in the drowsy_frames folder
    frame_path = os.path.join(drowsy_frames_folder, f"drowsy_{timestamp}.jpg")
    cv2.imwrite(frame_path, frame)

    # Wait for 5 seconds while playing the alert sound
    start_time = time.time()
    while time.time() - start_time < 5:
        continue

    # Stop the alert sound
    pygame.mixer.music.stop()

# Capture the video stream from the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

consecutive_drowsy_frames = 0
drowsy_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect drowsiness using YOLOv8
    results = model(frame)

    # Check if drowsiness is detected
    drowsiness_detected = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls)
            conf = box.conf.item()  # Convert tensor to scalar
            label = f"{class_names[cls]} {conf:.2f}"
            x1, y1, x2, y2 = box.xyxy.tolist()[0]  # Access individual elements directly
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to integers
            
            if class_names[cls] == "Drowsy":
                # Bounding box color for drowsy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                # Bounding box color for non-drowsy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

            if class_names[cls] == "Drowsy":
                drowsiness_detected = True

    if drowsiness_detected:
        if drowsy_start_time is None:
            drowsy_start_time = time.time()
        consecutive_drowsy_frames += 1

        if consecutive_drowsy_frames >= 90:  # 90 frames =  seconds at 30 FPS
            trigger_alert(frame)
            consecutive_drowsy_frames = 0
            drowsy_start_time = None
    else:
        consecutive_drowsy_frames = 0
        drowsy_start_time = None

    # Display the real-time video with bounding boxes
    cv2.imshow("Drowsiness Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Stop the pygame mixer
