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

# Lấy node D2, D4, M0, M1 để thực hiện ghi giá trị
D1_node = client.get_node("ns=2;s=Channel1.Device1.D1")
D2_node = client.get_node("ns=2;s=Channel1.Device1.D2")
D3_node = client.get_node("ns=2;s=Channel1.Device1.D3")
D4_node = client.get_node("ns=2;s=Channel1.Device1.D4")
M0_node = client.get_node("ns=2;s=Channel1.Device1.M0")
M1_node = client.get_node("ns=2;s=Channel1.Device1.M1")

# RTSP stream URL
rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"
cam = cv2.VideoCapture(0)

# Load YOLOv8 face detection model
model = YOLO(r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt").to('cuda')

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

    frame = cv2.resize(frame, (800, 600))
    results = model(frame, verbose=False)

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        print(f'Center Coordinates: ({center_x}, {center_y})')

        try:
            # trục 1
            M0_node.set_value(ua.DataValue(
                ua.Variant(True, ua.VariantType.Boolean)))
            D2_node.set_value(ua.DataValue(
                ua.Variant(center_x, ua.VariantType.UInt16)))
            D1_node.set_value(ua.DataValue(
                ua.Variant(400, ua.VariantType.UInt16)))
            # trục 2
            M1_node.set_value(ua.DataValue(
                ua.Variant(True, ua.VariantType.Boolean)))
            D4_node.set_value(ua.DataValue(
                ua.Variant(center_y, ua.VariantType.UInt16)))
            D3_node.set_value(ua.DataValue(
                ua.Variant(400, ua.VariantType.UInt16)))
        except Exception as e:
            print(f"Failed to write to OPC UA nodes: {e}")

        cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
        cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
        cv2.line(frame, (0, center_y), (800, center_y), (0, 0, 0), 2)
        cv2.line(frame, (center_x, 0), (center_x, 600), (0, 0, 0), 2)
    else:
        # Khi không có nhận diện khuôn mặt, vẽ ở trung tâm mặc định và đặt các giá trị về 0
        default_center_x, default_center_y = 400, 300
        cv2.circle(frame, (default_center_x, default_center_y),
                   60, (0, 0, 255), 2)
        cv2.circle(frame, (default_center_x, default_center_y),
                   15, (0, 0, 255), -1)
        cv2.line(frame, (0, default_center_y),
                 (800, default_center_y), (0, 0, 0), 2)
        cv2.line(frame, (default_center_x, 0),
                 (default_center_x, 600), (0, 0, 0), 2)

        try:
            # Đặt các giá trị về 0
            M0_node.set_value(ua.DataValue(
                ua.Variant(False, ua.VariantType.Boolean)))
            M1_node.set_value(ua.DataValue(
                ua.Variant(False, ua.VariantType.Boolean)))
            D2_node.set_value(ua.DataValue(
                ua.Variant(0, ua.VariantType.UInt16)))
            D4_node.set_value(ua.DataValue(
                ua.Variant(0, ua.VariantType.UInt16)))
        except Exception as e:
            print(f"Failed to reset OPC UA nodes: {e}")

    current_time = time.time()
    elapsed_time = current_time - prev_time
    fps = 1 / elapsed_time
    prev_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f'Faces: {1 if results and len(results[0].boxes) > 0 else 0}', (
        10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
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
