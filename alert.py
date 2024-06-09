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

# Create a folder to save screenshots
screenshot_dir = "screenshots"
if not os.path.exists(drowsy_frames_folder):
    os.makedirs(drowsy_frames_folder)

# Initialize dashboard variables
dashboard_frame = None
mute_alert = False
last_prediction_frame = None

# Function to trigger an alert and save the drowsy frame
def trigger_alert(frame):
    global mute_alert
    # Code to trigger an alert (e.g., play a sound, send a notification, etc.)
    print("Drowsiness detected for 3 seconds! Alert triggered.")
    logging.info("Drowsiness detected for 3 seconds")

    if not mute_alert:
        # Play a sound alert for 5 seconds on repeat
        pygame.mixer.init()
        pygame.mixer.music.load("AlertSound/beep-warning-6387.mp3")
        pygame.mixer.music.play(-1)  # -1 means loop indefinitely

    # Add the timestamp to the frame
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Date: {timestamp}"
    cv2.putText(frame, text, (10, 30), font, 0.5, (0, 0, 255), 1)

    # Save the frame with the timestamp in the drowsy_frames folder
    frame_path = os.path.join(drowsy_frames_folder, f"drowsy_{timestamp}.jpg")
    cv2.imwrite(frame_path, frame)

    # Update last prediction frame
    global last_prediction_frame
    last_prediction_frame = frame.copy()

    # Wait for 5 seconds while playing the alert sound
    start_time = time.time()
    while time.time() - start_time < 5:
        continue

    # Stop the alert sound
    if not mute_alert:
        pygame.mixer.music.stop()

# Function to handle button clicks
def handle_click(event, x, y, flags, param):
    global mute_alert, dashboard_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if the mute button is clicked
        if 20 <= x <= 160 and 440 <= y <= 480:
            mute_alert = not mute_alert
            print("Alert Muted" if mute_alert else "Alert Unmuted")
        # Check if the screenshot button is clicked
        elif 180 <= x <= 320 and 440 <= y <= 480:
            screenshot_path = os.path.join(screenshot_dir, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(screenshot_path, dashboard_frame)
            print(f"Screenshot saved as {screenshot_path}")
        # Check if the terminate button is clicked
        elif 340 <= x <= 480 and 440 <= y <= 480:
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            exit()

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
                drowsiness_detected = True
            else:
                # Bounding box color for non-drowsy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

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

    # Update dashboard frame
    dashboard_frame = frame.copy()

    # Draw buttons on the dashboard
    dashboard = dashboard_frame.copy()
    cv2.rectangle(dashboard, (20, 440), (160, 480), (100, 0, 0) if mute_alert else (0, 100, 0), -1)
    cv2.putText(dashboard, "Mute Alert", (30, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(dashboard, (180, 440), (320, 480), (255, 0, 0), -1)
    cv2.putText(dashboard, "Take Screenshot", (190, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.rectangle(dashboard, (340, 440), (480, 480), (0, 0, 255), -1)
    cv2.putText(dashboard, "Terminate", (350, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the dashboard
    cv2.imshow("Drowsiness Detection", dashboard)

    # Set mouse callback to handle button clicks
    cv2.setMouseCallback("Drowsiness Detection", handle_click)

    # Press 'q' to quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()  # Stop the pygame mixer

# Save the last prediction frame
if last_prediction_frame is not None:
    cv2.imwrite("last_prediction.jpg", last_prediction_frame)
    print("Last prediction screenshot saved as last_prediction.jpg")
