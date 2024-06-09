import cv2
import uuid
import os
import time

img_path = os.path.join('data', 'img')
labels = ['Non-Drowsy']
num_img = 25


capture = cv2.VideoCapture(0)

# Check if the directory exists, if not create it
if not os.path.exists(img_path):
    os.makedirs(img_path, exist_ok=True)

# Loop through labels
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(0.5)

    # Loop through images
    for img in range(num_img):
        print('Collecting images for {} number {}'.format(label, img))

        # Capture frame from the webcam
        ret, frame = capture.read()

        # Check if frame is not None
        if frame is not None:
            # Naming out image path
            image = os.path.join(img_path, label+'.'+str(uuid.uuid1())+'.jpg')
            print("Image path:", image)  # Print the image path for debugging


            cv2.imshow('Image Collection', frame)
            # Writes out image to file
            success = cv2.imwrite(image, frame)
            
            # Check if image was written successfully
            if success:
                print("Image saved successfully")
            else:
                print("Failed to save image")
                
            # Render to the screen
            
            
            # 15 second delay between captures
            time.sleep(0.5)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error: Frame is None")

capture.release()
cv2.destroyAllWindows()