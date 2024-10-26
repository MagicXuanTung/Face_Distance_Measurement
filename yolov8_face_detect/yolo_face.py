import cv2
import time
from ultralytics import YOLO


# RTSP stream URL
rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"

# Initialize webcam or RTSP stream
cam = cv2.VideoCapture(rtsp_url)

# Load YOLOv8 face detection model
model = YOLO(
    r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolo_face.py").to('cuda')

# Thời gian để tính FPS
prev_time = 0
max_fps = 100
min_time_between_frames = 1.0 / max_fps

# Main loop
while True:
    start_time = time.time()
    suc, frame = cam.read()
    if not suc:
        print("No image")
        break

    frame = cv2.resize(frame, (1280, 720))
    results = model(frame, verbose=False)

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        print(f'Center Coordinates: ({center_x}, {center_y})')
        cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
        cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
        cv2.line(frame, (0, center_y), (1280, center_y), (0, 0, 0), 2)
        cv2.line(frame, (center_x, 0), (center_x, 720), (0, 0, 0), 2)

    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time
    prev_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Faces: {1 if results and len(results[0].boxes) > 0 else 0}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Face Detection", frame)

    time_to_sleep = min_time_between_frames - (time.time() - start_time)
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
