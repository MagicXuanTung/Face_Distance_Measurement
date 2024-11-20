import cv2
import time
import threading
from ultralytics import YOLO
from opcua import Client, ua
import tkinter as tk
from tkinter import messagebox, filedialog  # Thêm filedialog

# --------------------------- Kết nối OPC UA ---------------------------
url = "opc.tcp://127.0.0.1:49320"
client = Client(url)

try:
    client.connect()
    print("Connected to Kepware OPC UA Server")
except Exception as e:
    print(f"Failed to connect to OPC UA Server: {e}")
    exit()

D1_node = client.get_node("ns=2;s=Channel1.Device1.D1")
D2_node = client.get_node("ns=2;s=Channel1.Device1.D2")
D4_node = client.get_node("ns=2;s=Channel1.Device1.D4")
D5_node = client.get_node("ns=2;s=Channel1.Device1.D5")

# Cập nhật giá trị ban đầu cho D1 và D5
try:
    D5_node.set_value(ua.DataValue(ua.Variant(6400, ua.VariantType.UInt16)))
    D1_node.set_value(ua.DataValue(ua.Variant(6400, ua.VariantType.UInt16)))
except Exception as e:
    print(f"Failed to set value for D1 and D5: {e}")

# --------------------------- Khởi tạo Camera và Mô hình YOLO ---------------------------
rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"
cam = cv2.VideoCapture(0)
model = YOLO(r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt").to('cuda')

# --------------------------- Hàm tiện ích ---------------------------


def clamp(value, min_value=0, max_value=65535):
    return max(min_value, min(value, max_value))


# --------------------------- Biến và cấu hình ---------------------------
width, height = 800, 600
prev_time = 0
is_face_detection_on = True
show_bounding_box = True
auto_capture = False
capture_directory = ""  # Biến lưu trữ thư mục chụp ảnh

# --------------------------- UI Setup with Tkinter ---------------------------
root = tk.Tk()
root.title("Camera Control Panel")

# Hàm chọn thư mục lưu ảnh


def choose_directory():
    global capture_directory
    capture_directory = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
    if capture_directory:
        print(f"Ảnh sẽ được lưu tại: {capture_directory}")
    else:
        print("Chưa chọn thư mục lưu ảnh")

# Các nút điều khiển


def toggle_face_detection():
    global is_face_detection_on
    is_face_detection_on = not is_face_detection_on
    status = "ON" if is_face_detection_on else "OFF"
    print(f"Face Detection: {status}")


def toggle_bounding_box():
    global show_bounding_box
    show_bounding_box = not show_bounding_box
    status = "Bounding Box" if show_bounding_box else "Border"
    print(f"Show: {status}")


def toggle_auto_capture():
    global auto_capture
    auto_capture = not auto_capture
    status = "ON" if auto_capture else "OFF"
    print(f"Auto Capture: {status}")


# Nút chọn thư mục lưu ảnh
btn_choose_directory = tk.Button(
    root, text="Chọn thư mục lưu ảnh", command=choose_directory)
btn_choose_directory.pack(pady=10)

btn_toggle_detection = tk.Button(
    root, text="Toggle Face Detection", command=toggle_face_detection)
btn_toggle_detection.pack(pady=10)

btn_toggle_bounding = tk.Button(
    root, text="Toggle Bounding Box/Border", command=toggle_bounding_box)
btn_toggle_bounding.pack(pady=10)

btn_toggle_capture = tk.Button(
    root, text="Toggle Auto Capture", command=toggle_auto_capture)
btn_toggle_capture.pack(pady=10)

# --------------------------- Hàm chính (vòng lặp OpenCV + Tkinter) ---------------------------


def process_frame():
    global prev_time, is_face_detection_on, show_bounding_box, auto_capture, capture_directory
    suc, frame = cam.read()  # Đọc một frame từ camera
    if not suc:
        print("No image")
        return

    frame = cv2.resize(frame, (width, height))  # Resize frame

    if is_face_detection_on:
        results = model(frame, verbose=False)

        center_x, center_y = width // 2, height // 2

        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            opc_center_x = clamp(center_x * 100)
            opc_center_y = clamp(center_y * 100)

            D2_node.set_value(ua.DataValue(ua.Variant(
                opc_center_x, ua.VariantType.UInt16)))
            D4_node.set_value(ua.DataValue(ua.Variant(
                opc_center_y, ua.VariantType.UInt16)))

            if show_bounding_box:
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), (0, 255, 0), 2)
            else:
                cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 0), 2)
                cv2.line(frame, (center_x, 0),
                         (center_x, height), (0, 0, 0), 2)
                cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
                cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
                cv2.putText(frame, f'Center: ({opc_center_x}, {opc_center_y})', (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "X+", (10, center_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "X-", (width - 30, center_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Hiển thị ký hiệu "Y+" và "Y-" cho trục Y
                cv2.putText(frame, "Y+", (center_x + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Y-", (center_x + 10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if auto_capture and capture_directory:  # Chỉ lưu nếu đã chọn thư mục
                timestamp = time.time()
                filename = f"{capture_directory}/captured_face_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Captured image at {filename}")

        else:
            D2_node.set_value(ua.DataValue(
                ua.Variant(0, ua.VariantType.UInt16)))
            D4_node.set_value(ua.DataValue(
                ua.Variant(0, ua.VariantType.UInt16)))

            cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
            cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 0), 2)
            cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 0), 2)
            cv2.putText(frame, "X+", (10, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "X-", (width - 30, center_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Y+", (center_x + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Y-", (center_x + 10, height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

# --------------------------- Vòng lặp chính ---------------------------


def loop():
    while True:
        process_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# Chạy GUI Tkinter và vòng lặp OpenCV
if __name__ == "__main__":
    threading.Thread(target=loop, daemon=True).start()
    root.mainloop()
