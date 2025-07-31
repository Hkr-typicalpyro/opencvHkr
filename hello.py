import cv2
import time
from cvzone.HandTrackingModule import HandDetector

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1080)  # Set width
cap.set(4, 720)   # Set height

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Get the start time
start_time = time.time()

while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break

    # Detect hands and draw them on the image
    hands, img = detector.findHands(img)

    # Show the image in a window
    cv2.imshow("Hand Tracking", img)

    # Break the loop after 10 seconds
    if time.time() - start_time > 10:
        break

    # Exit on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()