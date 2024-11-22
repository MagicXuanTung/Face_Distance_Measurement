import cv2
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# --------------------------- Khởi tạo Camera và Mô hình YOLO ---------------------------
# Mở camera (camera mặc định là 0)
cam = cv2.VideoCapture(0)

# Tải mô hình YOLO (đường dẫn đến file mô hình) và chuyển sang sử dụng GPU ('cuda')
model = YOLO(
    r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt").to('cuda')

# --------------------------- Biến và cấu hình ---------------------------
# Định nghĩa kích thước khung hình hiển thị
width, height = 800, 600

# Biến trạng thái để điều khiển các chế độ
is_face_detection_on = True  # Bật/Tắt nhận diện khuôn mặt
# Chế độ hiển thị bounding box (true: nhiều đối tượng, false: chỉ đối tượng duy nhất)
show_bounding_box = False
auto_capture = False  # Chế độ tự động lưu ảnh
capture_directory = ""  # Thư mục lưu ảnh
single_object_mode = False  # Chế độ nhận diện một đối tượng cụ thể

# --------------------------- Hàm chọn và mở thư mục ---------------------------

# Hàm chọn thư mục lưu ảnh


def choose_directory():
    global capture_directory
    # Hiển thị hộp thoại chọn thư mục
    capture_directory = filedialog.askdirectory(title="Chọn thư mục lưu ảnh")
    if capture_directory:
        print(f"Ảnh sẽ được lưu tại: {capture_directory}")
        # Nếu có thư mục, hiển thị nút mở thư mục
        btn_open_directory.pack(side=tk.LEFT, padx=5)
    else:
        print("Chưa chọn thư mục lưu ảnh")
        btn_open_directory.pack_forget()  # Nếu không chọn, ẩn nút mở thư mục

# Hàm mở thư mục đã chọn


def open_directory():
    if capture_directory:
        os.startfile(capture_directory)  # Mở thư mục trên hệ điều hành Windows
    else:
        print("Chưa chọn thư mục lưu ảnh")

# --------------------------- Các hàm bật/tắt các chế độ ---------------------------

# Bật/Tắt chế độ nhận diện khuôn mặt


def toggle_face_detection():
    global is_face_detection_on
    is_face_detection_on = not is_face_detection_on
    status = "ON" if is_face_detection_on else "OFF"
    print(f"Face Detection: {status}")
    update_button_state(btn_toggle_detection,
                        is_face_detection_on)  # Đổi màu nút

# Chuyển sang chế độ nhiều đối tượng (Bounding Box)


def toggle_multi_object_mode():
    global show_bounding_box
    show_bounding_box = True
    print("Mode: Multiple Objects Mode")

# Chuyển sang chế độ nhận diện một đối tượng duy nhất (Lock)


def toggle_single_object_mode():
    global show_bounding_box
    show_bounding_box = False
    print("Mode: Single Object Mode")

# Bật/Tắt chế độ tự động lưu ảnh


def toggle_auto_capture():
    global auto_capture
    auto_capture = not auto_capture
    status = "ON" if auto_capture else "OFF"
    print(f"Auto Capture: {status}")
    update_button_state(btn_toggle_capture, auto_capture)  # Đổi màu nút

# Cập nhật trạng thái nút (màu sắc tùy theo trạng thái bật/tắt)


def update_button_state(button, is_active):
    if is_active:
        button.config(bg="green", fg="white")  # Màu xanh khi bật
    else:
        # Màu mặc định khi tắt
        button.config(bg="SystemButtonFace", fg="black")


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

# Cập nhật trạng thái ban đầu cho nút
update_button_state(btn_toggle_capture, auto_capture)
update_button_state(btn_toggle_detection, is_face_detection_on)

# --------------------------- Hàm xử lý video ---------------------------

# Hàm giới hạn giá trị (clamp) trong khoảng min và max


def clamp(value, min_value=0, max_value=65535):
    return max(min_value, min(value, max_value))

# Hàm xử lý từng khung hình (frame)


def process_frame():
    prev_time = time.time()  # Lấy thời gian bắt đầu
    global is_face_detection_on, show_bounding_box, auto_capture, capture_directory, single_object_mode

    # Đọc khung hình từ camera
    suc, frame = cam.read()
    if not suc:
        return

    # Resize khung hình theo kích thước đã định
    frame = cv2.resize(frame, (width, height))

    # Khởi tạo tọa độ trung tâm mặc định
    center_x, center_y = width // 2, height // 2
    face_count = 0  # Đếm số khuôn mặt nhận diện được

    # Nếu nhận diện khuôn mặt được bật
    if is_face_detection_on:
        # Chạy mô hình YOLO trên khung hình
        results = model(frame, verbose=False)

        # Nếu phát hiện được đối tượng
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

        # Hiển thị số lượng đối tượng trên màn hình
        cv2.putText(frame, f"DOI TUONG: {face_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        if face_count == 0:
            cv2.putText(frame, "khong thay doi tuong", (width // 3, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Lưu khung hình tự động nếu bật chế độ tự động chụp và phát hiện được đối tượng
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

    return frame

# Hàm cập nhật khung hình liên tục trên giao diện


def update_frame():
    global canvas
    frame = process_frame()
    # Chuyển khung hình sang RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
    canvas.create_image(0, 0, image=photo, anchor=tk.NW)
    canvas.image = photo
    canvas.after(10, update_frame)  # Cập nhật lại sau 10ms


# Chạy vòng lặp cập nhật video
update_frame()
root.mainloop()
