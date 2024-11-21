import cv2
import time
from ultralytics import YOLO
from opcua_client import OPCUAClient  # Import module OPC UA

# --------------------------- Kết nối OPC UA ---------------------------

opcua_url = "opc.tcp://127.0.0.1:49320"
opcua_client = OPCUAClient(opcua_url)

# Kết nối OPC UA
opcua_client.connect()

# Lấy các node cần thiết
D1_node = opcua_client.get_node("ns=2;s=Channel1.Device1.D1")
D2_node = opcua_client.get_node("ns=2;s=Channel1.Device1.D2")
D4_node = opcua_client.get_node("ns=2;s=Channel1.Device1.D4")
D5_node = opcua_client.get_node("ns=2;s=Channel1.Device1.D5")

# Set giá trị ban đầu
opcua_client.set_value(D1_node, 6400)
opcua_client.set_value(D5_node, 6400)

# --------------------------- Khởi tạo Camera và Mô hình YOLO ---------------------------

cam = cv2.VideoCapture(0)
model = YOLO(r"C:\path\to\yolov8n-face.pt").to('cuda')

# --------------------------- Hàm tiện ích ---------------------------


def clamp(value, min_value=0, max_value=65535):
    return max(min_value, min(value, max_value))

# --------------------------- Vòng lặp chính ---------------------------


width, height = 800, 600
prev_time = 0

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
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        opc_center_x = clamp(center_x * 100)
        opc_center_y = clamp(center_y * 100)

        print(f"Tọa độ gửi qua OPC UA: ({opc_center_x}, {opc_center_y})")
        opcua_client.set_value(D2_node, opc_center_x)
        opcua_client.set_value(D4_node, opc_center_y)

        cv2.putText(frame, f'Center: ({opc_center_x}, {opc_center_y})', (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        opcua_client.set_value(D2_node, 0)
        opcua_client.set_value(D4_node, 0)

    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --------------------------- Đóng kết nối và giải phóng tài nguyên ---------------------------

opcua_client.disconnect()
cam.release()
cv2.destroyAllWindows()
