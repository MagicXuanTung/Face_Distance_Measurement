import cv2
import time
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --------------------------- Khởi tạo Camera và Mô hình YOLO ---------------------------
cam = cv2.VideoCapture(0)
model = YOLO(r"C:\Users\magic\Desktop\ĐỒ ÁN TỐT NGHIỆP\Face_Distance_Measurement\yolov8_face_detect\yolov8n-face.pt").to('cuda')

# --------------------------- Biến và cấu hình ---------------------------
prev_time = time.time()  # Initialize prev_time before the loop starts
width, height = 800, 600
show_axes = True
show_bounding_box_only = False

# --------------------------- Chức năng GUI ---------------------------


def toggle_axes():
    global show_axes
    show_axes = not show_axes
    status = "ON" if show_axes else "OFF"
    messagebox.showinfo("Trục tọa độ", f"Hiển thị trục tọa độ: {status}")


def toggle_bounding_box():
    global show_bounding_box_only
    show_bounding_box_only = not show_bounding_box_only
    status = "Chỉ bounding box" if show_bounding_box_only else "Đầy đủ"
    messagebox.showinfo("Bounding Box", f"Chế độ: {status}")

# --------------------------- Hiển thị GUI ---------------------------


def start_camera():
    global show_axes, show_bounding_box_only, prev_time  # Make sure prev_time is global
    prev_time = time.time()  # Initialize prev_time before the loop starts

    while True:
        suc, frame = cam.read()
        if not suc:
            print("No image")
            break

        frame = cv2.resize(frame, (width, height))
        results = model(frame, verbose=False)

        center_x, center_y = width // 2, height // 2
        if results and len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)

            if show_bounding_box_only:
                # Chỉ hiển thị bounding box mà không vẽ tâm
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f'Center: ({center_x}, {center_y})',
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not show_bounding_box_only and show_axes:
            # Vẽ trục tọa độ nếu không ở chế độ chỉ bounding box
            cv2.line(frame, (0, center_y), (width, center_y), (0, 0, 0), 2)
            cv2.line(frame, (center_x, 0), (center_x, height), (0, 0, 0), 2)
            cv2.circle(frame, (center_x, center_y), 60, (0, 0, 255), 2)
            cv2.circle(frame, (center_x, center_y), 15, (0, 0, 255), -1)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Chuyển đổi frame OpenCV thành ảnh Tkinter để hiển thị trên Canvas
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)

        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk

        # Cập nhật giao diện Tkinter
        root.update_idletasks()
        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# --------------------------- Tạo giao diện chính ---------------------------


root = tk.Tk()
root.title("Face Detection Control")
root.geometry("800x800")  # Đặt kích thước của cửa sổ

# Tạo Canvas để hiển thị video từ camera
canvas = tk.Canvas(root, width=width, height=height)
canvas.pack(pady=20)

# Tạo Frame chứa các nút điều khiển bên dưới
control_frame = tk.Frame(root)
control_frame.pack(side=tk.BOTTOM, pady=20)

# Nút điều khiển trục tọa độ
btn_toggle_axes = tk.Button(
    control_frame, text="Bật/Tắt Trục Tọa Độ", command=toggle_axes)
btn_toggle_axes.pack(pady=10)

# Nút điều khiển Bounding Box
btn_toggle_bounding_box = tk.Button(
    control_frame, text="Chuyển Bounding Box", command=toggle_bounding_box)
btn_toggle_bounding_box.pack(pady=10)

# Nút bắt đầu camera
btn_start_camera = tk.Button(
    control_frame, text="Bắt đầu Camera", command=start_camera)
btn_start_camera.pack(pady=10)

root.mainloop()
