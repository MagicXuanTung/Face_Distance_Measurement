import cv2
import time
from ultralytics import YOLO
from opcua import Client, ua

# Địa chỉ OPC UA của Kepware
url = "opc.tcp://127.0.0.1:49320"
client = Client(url)

# Kết nối đến máy chủ
try:
    client.connect()
    print("Connected to Kepware OPC UA Server")
except Exception as e:
    print(f"Failed to connect to OPC UA Server: {e}")
    exit()

# Lấy node D2, D3, D4, M0, M1 để thực hiện ghi giá trị
D2_node = client.get_node("ns=2;s=Channel1.Device1.D2")
D4_node = client.get_node("ns=2;s=Channel1.Device1.D4")
M0_node = client.get_node("ns=2;s=Channel1.Device1.M0")
M1_node = client.get_node("ns=2;s=Channel1.Device1.M1")

# Đặt giá trị cố định cho D1 và D3
D1_value = 400  # Thay đổi giá trị theo nhu cầu
D3_value = 400  # Thay đổi giá trị theo nhu cầu

# Ghi giá trị cố định cho D1 và D3
try:
    D1_node = client.get_node("ns=2;s=Channel1.Device1.D1")
    D3_node = client.get_node("ns=2;s=Channel1.Device1.D3")
    D1_node.set_value(ua.DataValue(ua.Variant(D1_value, ua.VariantType.Int32)))
    D3_node.set_value(ua.DataValue(ua.Variant(D3_value, ua.VariantType.Int32)))
except Exception as e:
    print(f"Failed to set fixed values for D1 and D3: {e}")

# RTSP stream URL
rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"
cam = cv2.VideoCapture(0)

# Load YOLOv8 face detection model
model = YOLO(r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt").to('cuda')

# Thời gian để tính FPS
prev_time = 0
max_fps = 100
min_time_between_frames = 1.0 / max_fps

# Kích thước khung hình
width, height = 800, 600


# Main loop
while True:
    start_time = time.time()
    suc, frame = cam.read()
    if not suc:
        print("No image")
        break

    frame = cv2.resize(frame, (width, height))
    results = model(frame, verbose=False)

    # Vẽ hệ trục tọa độ
    cv2.line(frame, (width // 2, 0), (width // 2, height),
             (255, 0, 0), 2)  # Trục y
    cv2.line(frame, (0, height // 2), (width, height // 2),
             (255, 0, 0), 2)  # Trục x
    cv2.putText(frame, "X", (width - 30, height // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Y", (width // 2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        # Chuyển đổi tọa độ
        adjusted_center_x = center_x - (width // 2)
        adjusted_center_y = (height // 2) - center_y

        # Kiểm tra và đặt giá trị âm
        adjusted_center_x = adjusted_center_x if adjusted_center_x != 0 else -1
        adjusted_center_y = adjusted_center_y if adjusted_center_y != 0 else -1

        print(
            f'Adjusted Center Coordinates: ({adjusted_center_x}, {adjusted_center_y})')

        try:
            M0_node.set_value(ua.DataValue(
                ua.Variant(True, ua.VariantType.Boolean)))
            D2_node.set_value(ua.DataValue(ua.Variant(
                adjusted_center_x, ua.VariantType.Int32)))
            M1_node.set_value(ua.DataValue(
                ua.Variant(True, ua.VariantType.Boolean)))
            D4_node.set_value(ua.DataValue(ua.Variant(
                adjusted_center_y, ua.VariantType.Int32)))
        except Exception as e:
            print(f"Failed to write to OPC UA nodes: {e}")

        # Vẽ đường thẳng từ tâm hình tròn
        cv2.line(frame, (center_x, center_y), (center_x, 0),
                 (0, 0, 0), 2)  # Đường thẳng dọc lên trên
        cv2.line(frame, (center_x, center_y), (center_x, height),
                 (0, 0, 0), 2)  # Đường thẳng dọc xuống dưới
        cv2.line(frame, (center_x, center_y), (0, center_y),
                 (0, 0, 0), 2)  # Đường thẳng ngang sang trái
        cv2.line(frame, (center_x, center_y), (width, center_y),
                 (0, 0, 0), 2)  # Đường thẳng ngang sang phải

        # Vẽ các hình tròn
        cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
        cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)

    else:
        # Nếu không có khuôn mặt, vẽ ở trung tâm mặc định
        default_center_x, default_center_y = width // 2, height // 2
        cv2.circle(frame, (default_center_x, default_center_y),
                   60, (0, 0, 255), 2)
        cv2.circle(frame, (default_center_x, default_center_y),
                   15, (0, 0, 255), -1)

        try:
            M0_node.set_value(ua.DataValue(
                ua.Variant(False, ua.VariantType.Boolean)))
            M1_node.set_value(ua.DataValue(
                ua.Variant(False, ua.VariantType.Boolean)))
            D2_node.set_value(ua.DataValue(
                ua.Variant(0, ua.VariantType.Int32)))
            D4_node.set_value(ua.DataValue(
                ua.Variant(0, ua.VariantType.Int32)))
        except Exception as e:
            print(f"Failed to reset OPC UA nodes: {e}")

    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time
    prev_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f'Faces: {1 if results and len(results[0].boxes) > 0 else 0}', (
        10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.imshow("Face Detection", frame)

    time_to_sleep = min_time_between_frames - (time.time() - start_time)
    if time_to_sleep > 0:
        time.sleep(time_to_sleep)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng kết nối và giải phóng tài nguyên
client.disconnect()
cam.release()
cv2.destroyAllWindows()
