import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 480)
cam.set(4, 640)
cam.set(10, 100)

# Initialize face mesh detector
detector = FaceMeshDetector(maxFaces=1)

# Texts to display
texts = [
    'Font size changes',
    'depending on distance',
    'from face to camera'
]
sen = 10

# Main loop
while True:
    suc, frame = cam.read()
    text_img = np.zeros_like(frame)
    if not suc:
        print("No image")
        break
    img, faces = detector.findFaceMesh(frame, draw=True)
    if faces:
        face = faces[0]
        left = face[145]
        right = face[374]
        W = 6.3  # Average distance between eyes (in cm)
        w, _, _ = detector.findDistance(right, left, img)
        f = 132.28346456692915  # Focal length

        d = (W * f) / w  # Calculate distance
        cvzone.putTextRect(
            img, f'Distance {int(d)} cm', (face[10][1] + 100, face[10][1] - 30), 2, 1)

        # Calculate bounding box coordinates
        x_coords = [p[0] for p in face]
        y_coords = [p[1] for p in face]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Draw bounding box
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        for i, text in enumerate(texts):
            height = 30 + int((int(d / sen) * sen) / 4)
            scale = 0.3 + (int(d / sen) * 10) / 100
            cv2.putText(text_img, text, (50, 50 + (i * height)),
                        cv2.FONT_HERSHEY_TRIPLEX, scale, (255, 255, 255), 2)

    stack_img = cvzone.stackImages([img, text_img], 2, 1)
    cv2.imshow("Dynamic_Text_Reader", stack_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
