import cv2
import time
from ultralytics import YOLO

# RTSP stream URL
rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"

# Initialize webcam or RTSP stream
cam = cv2.VideoCapture(rtsp_url)

# Load YOLOv8 face detection model
# cuda for GPU
# cpu for CPU
model = YOLO(
    r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8n-face.pt").to('cuda')

# Thời gian để tính FPS
prev_time = 0

# Optional: Giới hạn FPS để giảm tải (giới hạn tối đa 15 FPS)
max_fps = 100
min_time_between_frames = 1.0 / max_fps

# Main loop
while True:
    # Đo thời gian bắt đầu
    start_time = time.time()

    suc, frame = cam.read()
    if not suc:
        print("No image")
        break

    # Resize the frame (Giảm kích thước để tăng tốc độ)
    frame = cv2.resize(frame, (480, 360))

    # Dùng model YOLO để dự đoán khuôn mặt
    results = model(frame, verbose=False)  # Bỏ qua logging để tăng tốc

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]  # Lấy khuôn mặt đầu tiên

        # Lấy tọa độ của bounding box
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

        # Tính toán tọa độ trung tâm
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        cv2.putText(frame, f"Center: ({center_x}, {center_y})",
                    (center_x + 15, center_y - 15),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        # Vẽ bounding box và trung tâm
        cv2.circle(frame, (center_x, center_y), 35, (0, 0, 255), 2)
        cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)

        # Log tọa độ của điểm chính giữa
        print(f'Center Coordinates: ({center_x}, {center_y})')

        # Vẽ các đường thẳng theo trục x và y
        cv2.line(frame, (0, center_y), (480, center_y),
                 (0, 0, 0), 2)  # Đường x
        cv2.line(frame, (center_x, 0), (center_x, 360),
                 (0, 0, 0), 2)  # Đường y

    # Tính toán FPS
    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time
    prev_time = current_time

    # Hiển thị số lượng khuôn mặt và FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Faces: {1 if results and len(results[0].boxes) > 0 else 0}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Face Detection", frame)

    # Giới hạn tốc độ khung hình (FPS)
    time_to_sleep = min_time_between_frames - (time.time() - start_time)
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
