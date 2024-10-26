from opcua import Client, ua

# Địa chỉ OPC UA của Kepware
url = "opc.tcp://127.0.0.1:49320"
client = Client(url)

# try:
# Kết nối đến máy chủ
client.connect()
print("Connected to Kepware OPC UA Server")

# Kiểm tra session timeout để xác nhận
actual_timeout = client.session_timeout
print(
    f"Requested session timeout: 3600000ms, actual timeout: {actual_timeout}ms")

# Lấy node của X0 để thực hiện ghi giá trị
x0_node = client.get_node("ns=2;s=Channel1.Device1.X0")
write_value = False  # hoặc False nếu cần
x0_node.set_value(ua.DataValue(ua.Variant(
    write_value, ua.VariantType.Boolean)))
print(f"Value written to X0: {write_value}")

# Đọc giá trị của node D100
# d100_node = client.get_node(
#     "ns=2;s=Channel1.Device1.D100")  # Chỉ định node cho D100
# d100_value = d100_node.get_value()
# print("Current value of D100:", d100_value)

# except ua.UaStatusCodeError as e:
#     print("Lỗi khi đọc/ghi giá trị từ node:", e)
# except Exception as e:
#     print("Lỗi khi kết nối hoặc đọc/ghi dữ liệu:", e)

# finally:
#     # Ngắt kết nối
#     client.disconnect()
#     print("Disconnected from Kepware OPC UA Server")
