import cv2
import time
from ultralytics import YOLO
from opcua import Client, ua

# --------------------------- Kết nối OPC UA ---------------------------

# Địa chỉ OPC UA của Kepware
url = "opc.tcp://127.0.0.1:49320"
client = Client(url)  # Tạo đối tượng Client để kết nối đến máy chủ OPC UA

# Kết nối đến máy chủ OPC UA
try:
    client.connect()
    # Thông báo nếu kết nối thành công
    print("Connected to Kepware OPC UA Server")
except Exception as e:
    # Thông báo nếu kết nối thất bại
    print(f"Failed to connect to OPC UA Server: {e}")
    exit()  # Dừng chương trình nếu không kết nối được

# Lấy các node OPC UA cần thiết
D1_node = client.get_node("ns=2;s=Channel1.Device1.D1")
D2_node = client.get_node("ns=2;s=Channel1.Device1.D2")
D4_node = client.get_node("ns=2;s=Channel1.Device1.D4")
D5_node = client.get_node("ns=2;s=Channel1.Device1.D5")

# Set giá trị ban đầu cho D1 và D5
try:
    # Ghi giá trị 6400 vào D5
    D5_node.set_value(ua.DataValue(ua.Variant(6400, ua.VariantType.UInt16)))
    # Ghi giá trị 6400 vào D1
    D1_node.set_value(ua.DataValue(ua.Variant(6400, ua.VariantType.UInt16)))
except Exception as e:
    # Thông báo nếu có lỗi khi ghi giá trị
    print(f"Failed to set value for D1 and D5: {e}")

# --------------------------- Khởi tạo Camera và Mô hình YOLO ---------------------------

# Địa chỉ RTSP của camera
cam = cv2.VideoCapture(0)  # Mở camera hoặc stream RTSP

# Tải mô hình YOLOv8 để phát hiện khuôn mặt
# Chạy mô hình YOLO trên GPU
model = YOLO(r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt").to('cuda')

# --------------------------- Hàm tiện ích ---------------------------

# Hàm clamp để giới hạn giá trị trong phạm vi uint16 (0 - 65535)


def clamp(value, min_value=0, max_value=65535):
    return max(min_value, min(value, max_value))

# --------------------------- Biến và cấu hình ---------------------------


# Thời gian để tính FPS
prev_time = 0  # Biến để lưu thời gian trước khi tính FPS

# Kích thước cửa sổ hiển thị
width, height = 800, 600  # Kích thước của cửa sổ hiển thị hình ảnh

# --------------------------- Vòng lặp chính ---------------------------

while True:
    suc, frame = cam.read()  # Đọc một frame từ camera
    if not suc:  # Nếu không đọc được frame
        print("No image")  # In thông báo lỗi
        break  # Thoát vòng lặp nếu không có hình ảnh

    # Resize frame về kích thước cố định
    frame = cv2.resize(frame, (width, height))
    # Sử dụng YOLO để phát hiện khuôn mặt trong frame
    results = model(frame, verbose=False)

    # Mặc định tọa độ trung tâm là giữa màn hình
    center_x = width // 2
    center_y = height // 2

    if results and len(results[0].boxes) > 0:  # Nếu phát hiện khuôn mặt
        # Lấy box đầu tiên (khuôn mặt đầu tiên phát hiện được)
        box = results[0].boxes[0]
        x_min, y_min, x_max, y_max = map(
            int, box.xyxy[0])  # Lấy tọa độ của box

        # Tính tọa độ trung tâm của khuôn mặt
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)

        # Clamp giá trị tọa độ trung tâm trước khi gửi
        opc_center_x = clamp(center_x * 100)  # Nhân với 100 và clamp giá trị
        opc_center_y = clamp(center_y * 100)  # Nhân với 100 và clamp giá trị

        # In tọa độ trung tâm khuôn mặt
        print(f"Tọa độ pixel trung tâm khuôn mặt: ({center_x}, {center_y})")
        print(f"Tọa độ gửi qua OPC UA: ({opc_center_x}, {opc_center_y})")

        # Ghi giá trị vào các node OPC UA
        # Ghi tọa độ X vào D2
        D2_node.set_value(ua.DataValue(ua.Variant(
            opc_center_x, ua.VariantType.UInt16)))
        # Ghi tọa độ Y vào D4
        D4_node.set_value(ua.DataValue(ua.Variant(
            opc_center_y, ua.VariantType.UInt16)))

        # Hiển thị tọa độ trung tâm khuôn mặt lên frame
        cv2.putText(frame, f'Center: ({opc_center_x}, {opc_center_y})', (center_x + 10, center_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # In tọa độ lên hình ảnh
    else:
        # Nếu không phát hiện khuôn mặt, đặt giá trị mặc định là 0
        # Ghi giá trị 0 vào D2
        D2_node.set_value(ua.DataValue(ua.Variant(0, ua.VariantType.UInt16)))
        # Ghi giá trị 0 vào D4
        D4_node.set_value(ua.DataValue(ua.Variant(0, ua.VariantType.UInt16)))

    # Vẽ các đường và hình tròn vào màn hình để biểu diễn tọa độ trung tâm
    cv2.line(frame, (0, center_y), (width, center_y),
             (0, 0, 0), 2)  # Vẽ đường ngang qua trung tâm
    cv2.line(frame, (center_x, 0), (center_x, height),
             (0, 0, 0), 2)  # Vẽ đường dọc qua trung tâm
    # Vẽ vòng tròn lớn tại trung tâm
    cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
    # Vẽ vòng tròn nhỏ tại trung tâm
    cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)

    # Hiển thị ký hiệu "X+" và "X-" cho trục X
    cv2.putText(frame, "X+", (10, center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "X-", (width - 30, center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # Hiển thị ký hiệu "Y+" và "Y-" cho trục Y
    cv2.putText(frame, "Y+", (center_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Y-", (center_x + 10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tính toán FPS và hiển thị
    current_time = time.time()  # Lấy thời gian hiện tại
    fps = 1 / (current_time - prev_time)  # Tính FPS
    prev_time = current_time  # Cập nhật thời gian trước đó

    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)  # Hiển thị FPS lên màn hình

    # Hiển thị hình ảnh với kết quả lên cửa sổ
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Kiểm tra nếu người dùng nhấn 'q' để thoát
        break

# --------------------------- Đóng kết nối và giải phóng tài nguyên ---------------------------

client.disconnect()  # Ngắt kết nối OPC UA
cam.release()  # Giải phóng tài nguyên camera
cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ hi
