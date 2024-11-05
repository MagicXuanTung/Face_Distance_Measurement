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

# Lấy node D1, D2, D5, D4 để thực hiện ghi giá trị
D1_node = client.get_node("ns=2;s=Channel1.Device1.D1")
D2_node = client.get_node("ns=2;s=Channel1.Device1.D2")
D4_node = client.get_node("ns=2;s=Channel1.Device1.D4")
D5_node = client.get_node("ns=2;s=Channel1.Device1.D5")

# Set giá trị cho D1 và D5 là 6400
try:
    D5_node.set_value(ua.DataValue(ua.Variant(6400, ua.VariantType.UInt16)))
    D1_node.set_value(ua.DataValue(ua.Variant(6400, ua.VariantType.UInt16)))
except Exception as e:
    print(f"Failed to set value for D1 and D5: {e}")

# RTSP stream URL
rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"
cam = cv2.VideoCapture(0)

# Load YOLOv8 face detection model
model = YOLO(r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt").to('cuda')

# Thời gian để tính FPS
prev_time = 0

width, height = 800, 600

# Clamp function to ensure values are within ushort range


def clamp(value, min_value=0, max_value=65535):
    return max(min_value, min(value, max_value))


# Main loop
while True:
    suc, frame = cam.read()
    if not suc:
        print("No image")
        break

    frame = cv2.resize(frame, (width, height))
    results = model(frame, verbose=False)

    center_x = width // 2
    center_y = height // 2

    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

        # Tính tọa độ trung tâm khuôn mặt
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        # Clamp values before sending
        opc_center_x = clamp(center_x * 100)
        opc_center_y = clamp(center_y * 100)

        # Log tọa độ trung tâm khuôn mặt
        print(f"Tọa độ pixel trung tâm khuôn mặt: ({center_x}, {center_y})")
        print(f"Tọa độ gửi qua OPC UA: ({opc_center_x}, {opc_center_y})")

        # Ghi giá trị vào các node OPC UA
        D2_node.set_value(ua.DataValue(ua.Variant(
            opc_center_x, ua.VariantType.UInt16)))
        D4_node.set_value(ua.DataValue(ua.Variant(
            opc_center_y, ua.VariantType.UInt16)))

        # Hiển thị tọa độ trung tâm khuôn mặt
        cv2.putText(frame, f'Center: ({opc_center_x}, {opc_center_y})', (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # Nếu không phát hiện khuôn mặt, vẽ các đường và hình tròn ở giữa màn hình
        opc_center_x = clamp(center_x * 100)
        opc_center_y = clamp(center_y * 100)

        # Log tọa độ khi không phát hiện khuôn mặt
        print(f"Tọa độ trung tâm màn hình: ({center_x}, {center_y})")
        print(f"Tọa độ gửi qua OPC UA: ({opc_center_x}, {opc_center_y})")

        # Ghi giá trị vào các node OPC UA
        D2_node.set_value(ua.DataValue(ua.Variant(
            opc_center_x, ua.VariantType.UInt16)))
        D4_node.set_value(ua.DataValue(ua.Variant(
            opc_center_y, ua.VariantType.UInt16)))

    # Vẽ đường thẳng và hình tròn vào tọa độ trung tâm
    cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 0), 2)
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 0), 2)
    cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
    cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
    cv2.putText(frame, "X", (width - 30, center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(frame, "Y", (center_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Tính toán và hiển thị FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Đóng kết nối và giải phóng tài nguyên
client.disconnect()
cam.release()
cv2.destroyAllWindows()
