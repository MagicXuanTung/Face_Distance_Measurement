import asyncio
import tempfile
import threading
import cv2
import time
from telegram import Bot
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from opcua_client import OPCUAClient

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

try:
    # Ghi giá trị 6400 vào D5
    opcua_client.set_value(D1_node, 6400)
    opcua_client.set_value(D5_node, 6400)
except Exception as e:
    # Thông báo nếu có lỗi khi ghi giá trị
    print(f"Failed to set value for D1 and D5: {e}")


def clamp(value, min_value=0, max_value=65535):
    return max(min_value, min(value, max_value))


# --------------------------- Khởi tạo Camera và Mô hình YOLO ---------------------------
# Mở camera (camera mặc định là 0)
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise Exception("Không thể khởi động camera. Kiểm tra kết nối thiết bị!")

# Tải mô hình YOLO và chuyển sang sử dụng GPU ('cuda')
model_path = r"D:\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt"
model = YOLO(model_path).to('cuda')

# --------------------------- Biến và cấu hình ---------------------------
width, height = 800, 600
is_face_detection_on = True
show_bounding_box = False
auto_capture = False
capture_directory = ""
is_message_sending_on = False
single_object_mode = False
last_message_time = 0
MESSAGE_DELAY = 2

# Telegram
TOKEN = '7722510055:AAGW2PSNYdClv69vE0Rnj92StKP500xFeOE'
CHAT_ID = '-4755156196'

# --------------------------- Hàm xử lý ---------------------------


def toggle_multi_object_mode():
    global show_bounding_box
    show_bounding_box = True
    print("Mode: Multiple Objects Mode")


async def send_message(token, chat_id, message, photo_path):
    """Gửi tin nhắn kèm ảnh qua Telegram."""
    bot = Bot(token=token)
    try:
        with open(photo_path, 'rb') as photo_file:
            await bot.send_photo(chat_id=chat_id, photo=photo_file, caption=message)
    except Exception as e:
        print(f"Lỗi khi gửi tin nhắn Telegram: {e}")


def send_image_with_message(frame, message="Tin nhắn từ hệ thống"):
    """Lưu khung hình tạm thời và gửi tin nhắn Telegram."""
    global last_message_time

    # Kiểm tra nếu chưa đủ thời gian chờ giữa các lần gửi
    current_time = time.time()
    if current_time - last_message_time < MESSAGE_DELAY:
        print("Đang chờ để gửi tin nhắn tiếp theo...")
        return

    # Lưu lại thời gian gửi tin nhắn
    last_message_time = current_time

    # Lưu khung hình tạm thời
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.close()
        cv2.imwrite(temp_file.name, frame)

    def send_in_thread():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send_message(
                TOKEN, CHAT_ID, message, temp_file.name))
        finally:
            try:
                os.remove(temp_file.name)
            except Exception as e:
                print(f"Không thể xóa file tạm: {e}")

    threading.Thread(target=send_in_thread).start()


def clamp(value, min_value=0, max_value=65535):
    return max(min_value, min(value, max_value))


def toggle_single_object_mode():
    global show_bounding_box
    show_bounding_box = False
    print("Mode: Single Object Mode")


def toggle_message_sending():
    """Bật/Tắt chế độ gửi tin nhắn Telegram."""
    global is_message_sending_on
    is_message_sending_on = not is_message_sending_on
    update_button_state(btn_toggle_message, is_message_sending_on)


def choose_directory():
    """Chọn thư mục lưu ảnh."""
    global capture_directory
    capture_directory = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
    if capture_directory:
        print(f"Thư mục lưu ảnh: {capture_directory}")
        btn_open_directory.pack(side=tk.LEFT, padx=5)
    else:
        btn_open_directory.pack_forget()


def open_directory():
    """Mở thư mục lưu ảnh."""
    if capture_directory:
        os.startfile(capture_directory)
    else:
        print("Chưa chọn thư mục!")


def toggle_face_detection():
    """Bật/Tắt chế độ nhận diện khuôn mặt."""
    global is_face_detection_on
    is_face_detection_on = not is_face_detection_on
    update_button_state(btn_toggle_detection, is_face_detection_on)


def toggle_auto_capture():
    """Bật/Tắt chế độ tự động lưu ảnh."""
    global auto_capture
    auto_capture = not auto_capture
    update_button_state(btn_toggle_capture, auto_capture)


def update_button_state(button, is_active):
    """Cập nhật màu sắc nút dựa vào trạng thái."""
    if is_active:
        button.config(bg="green", fg="white")
    else:
        button.config(bg="SystemButtonFace", fg="black")


def process_frame():
    """Xử lý khung hình từ camera."""
    prev_time = time.time()
    global is_face_detection_on, auto_capture, is_message_sending_on, capture_directory, single_object_mode, show_bounding_box

    ret, frame = cam.read()
    if not ret:
        print("Không thể đọc dữ liệu từ camera.")
        return

    frame = cv2.resize(frame, (width, height))
    face_count = 0

    if is_face_detection_on:
        results = model(frame, verbose=False)
        if results and len(results[0].boxes) > 0:
            if show_bounding_box:  # Chế độ nhiều đối tượng
                for box in results[0].boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    center_x = int((x_min + x_max) / 2)
                    center_y = int((y_min + y_max) / 2)

                    opc_center_x = clamp(center_x * 100)
                    opc_center_y = clamp(center_y * 100)

                    class_id = int(box.cls[0])  # ID của class
                    class_name = model.names[class_id]
                    cv2.putText(frame, class_name, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x_min, y_min),
                                  (x_max, y_max), (0, 255, 0), 2)
                    face_count += 1

            elif not show_bounding_box:  # Chế độ một đối tượng
                box = results[0].boxes[0]
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)

                # Hiển thị tọa độ trung tâm và vẽ các đường/circle
                opc_center_x = clamp(center_x * 100)
                opc_center_y = clamp(center_y * 100)

                print(
                    f"Tọa độ gửi qua OPC UA: ({opc_center_x}, {opc_center_y})")
                opcua_client.set_value(D2_node, opc_center_x)
                opcua_client.set_value(D4_node, opc_center_y)

                cv2.putText(frame, f'Center: ({opc_center_x}, {opc_center_y})', (center_x + 10, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 0), 2)
                cv2.line(frame, (center_x, 0),
                         (center_x, height), (0, 0, 0), 2)
                cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
                cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)
                cv2.putText(frame, "X+", (10, center_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "X-", (width - 30, center_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Y+", (center_x + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, "Y-", (center_x + 10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                face_count += 1

            if is_message_sending_on and face_count > 0:
                send_image_with_message(frame, "Đã phát hiện khuôn mặt!")

        cv2.putText(frame, f"DOI TUONG: {face_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if face_count == 0:
            opcua_client.set_value(D2_node, 0)
            opcua_client.set_value(D4_node, 0)
            cv2.putText(frame, "khong thay doi tuong", (width // 3, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if auto_capture and face_count > 0 and capture_directory:
        timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
        capture_path = f"{capture_directory}/capture_{timestamp}.png"
        cv2.imwrite(capture_path, frame)
        print(f"Đã lưu ảnh tại: {capture_path}")

    # Tính FPS và hiển thị trên khung hình
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.imgtk = imgtk
    root.after(10, process_frame)


def on_closing():
    """Xử lý khi đóng ứng dụng."""
    if messagebox.askokcancel("Thoát", "Bạn có chắc muốn thoát?"):
        cam.release()
        root.destroy()


# --------------------------- UI Setup với Tkinter ---------------------------
# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("Camera Nhận Diện Đối Tượng")

# Canvas để hiển thị video
canvas = tk.Canvas(root, width=width, height=height)
canvas.pack()

# Frame chứa các nút điều khiển
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, pady=10)

# Tạo các nút điều khiển
btn_toggle_detection = tk.Button(
    button_frame, text="Bật/Tắt nhận diện", command=toggle_face_detection)
btn_toggle_detection.pack(side=tk.LEFT, padx=5)

btn_toggle_bounding = tk.Button(
    button_frame, text="Chế độ nhiều đối tượng (Bounding Box)", command=toggle_multi_object_mode)
btn_toggle_bounding.pack(side=tk.LEFT, padx=5)

btn_toggle_single_object_mode = tk.Button(
    button_frame, text="Chế độ 1 đối tượng (Lock)", command=toggle_single_object_mode)
btn_toggle_single_object_mode.pack(side=tk.LEFT, padx=5)

btn_choose_directory = tk.Button(
    button_frame, text="Chọn thư mục lưu ảnh", command=choose_directory)
btn_choose_directory.pack(side=tk.LEFT, padx=5)

btn_open_directory = tk.Button(
    button_frame, text="Mở thư mục lưu ảnh", command=open_directory)

btn_toggle_capture = tk.Button(
    button_frame, text="Bật/Tắt lưu ảnh", command=toggle_auto_capture)
btn_toggle_capture.pack(side=tk.LEFT, padx=5)

btn_toggle_message = tk.Button(
    button_frame, text="Bật/Tắt gửi tin nhắn", command=toggle_message_sending)
btn_toggle_message.pack(side=tk.LEFT, padx=5)


# Cập nhật trạng thái ban đầu cho nút
update_button_state(btn_toggle_capture, auto_capture)
update_button_state(btn_toggle_detection, is_face_detection_on)
update_button_state(btn_toggle_message, is_message_sending_on)

root.protocol("WM_DELETE_WINDOW", on_closing)

# --------------------------- Chạy ứng dụng ---------------------------
process_frame()
root.mainloop()
