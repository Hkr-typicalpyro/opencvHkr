import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# -------- CONFIG --------
image_path = 'C:\\Users\\rko_n\\Desktop\\opencvHkr\\download (1).png'  # 🔁 Change to your image file path
cam_width, cam_height = 1280, 720
min_distance = 50
max_distance = 300
min_scale = 100
max_scale = 500

# -------- SETUP --------
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = HandDetector(detectionCon=0.8, maxHands=2)

# Load the image
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Allows for PNG transparency

# Initial scale
scale = 300

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Detect hands
    hands, frame = detector.findHands(frame)

    if len(hands) == 2:
        hand1, hand2 = hands[0], hands[1]

        # Get center points of both hands
        lmList1 = hand1["lmList"]
        lmList2 = hand2["lmList"]

        # Check gesture: only thumb and index finger up
        fingers1 = detector.fingersUp(hand1)
        fingers2 = detector.fingersUp(hand2)

        if fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]:
            # Get center of both hands
            cx1, cy1 = hand1["center"]
            cx2, cy2 = hand2["center"]

            # Calculate distance
            distance = int(np.linalg.norm(np.array([cx2, cy2]) - np.array([cx1, cy1])))

            # Map distance to scale
            scale = np.interp(distance, [min_distance, max_distance], [min_scale, max_scale])
            scale = int(scale)

            # Resize image
            if img is not None:
                resized_img = cv2.resize(img, (scale, scale), interpolation=cv2.INTER_AREA)

                # Calculate position to overlay (midpoint between hands)
                mid_x = (cx1 + cx2) // 2 - scale // 2
                mid_y = (cy1 + cy2) // 2 - scale // 2

                # Overlay with transparency support
                h, w, _ = resized_img.shape
                if 0 <= mid_x <= cam_width - w and 0 <= mid_y <= cam_height - h:
                    alpha_img = resized_img[:, :, 3] / 255.0
                    alpha_frame = 1.0 - alpha_img

                    for c in range(3):
                        frame[mid_y:mid_y + h, mid_x:mid_x + w, c] = (alpha_img * resized_img[:, :, c] +
                                                                      alpha_frame * frame[mid_y:mid_y + h, mid_x:mid_x + w, c])

    # Show output
    cv2.imshow("Image Resizer", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
