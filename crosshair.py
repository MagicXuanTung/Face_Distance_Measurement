import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import time

rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"

# Initialize webcam
cam = cv2.VideoCapture(rtsp_url)

# Initialize face mesh detector
detector = FaceMeshDetector(maxFaces=1)

# Thời gian để tính FPS
prev_time = 0

# Main loop
while True:
    suc, frame = cam.read()
    if not suc:
        print("No image")
        break

    # Resize the frame
    frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

    # Không vẽ các điểm trên khuôn mặt
    img, faces = detector.findFaceMesh(frame, draw=False)  # Chuyển draw=False
    if faces:
        for face in faces:  # Loop through detected faces
            # Tính toán tọa độ bounding box
            x_coords = [p[0] for p in face]
            y_coords = [p[1] for p in face]
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)

            # Tính toán tọa độ trung tâm
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            # Vẽ vòng tròn quanh khuôn mặt
            cv2.circle(img, (center_x, center_y), 80,
                       (0, 0, 255), 2)  # Vòng tròn lớn
            # Vòng tròn nhỏ bên trong
            cv2.circle(img, (center_x, center_y), 15, (0, 0, 255), -1)

            # Vẽ văn bản hiển thị tọa độ trung tâm
            cv2.putText(img, f"Center: ({center_x}, {center_y})",
                        (center_x + 15, center_y - 15),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Vẽ các đường thẳng theo trục x và y
            cv2.line(img, (0, center_y), (640, center_y),
                     (0, 0, 0), 2)  # Đường x
            cv2.line(img, (center_x, 0), (center_x, 480),
                     (0, 0, 0), 2)  # Đường y

            # Log tọa độ của điểm chính giữa
            print(f'Center Coordinates: ({center_x}, {center_y})')

    # Tính toán FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Hiển thị số lượng khuôn mặt và FPS
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f'Faces: {len(faces)}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
