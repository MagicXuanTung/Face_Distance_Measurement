from pymodbus.client import ModbusTcpClient

# Thay đổi địa chỉ IP và cổng phù hợp với PLC của bạn
PLC_IP = '192.168.1.10'
PLC_PORT = 502

# Tạo client
client = ModbusTcpClient(PLC_IP, port=PLC_PORT)

# Kết nối
if client.connect():
    print("Kết nối thành công!")

    # Gửi dữ liệu (Ví dụ: ghi một giá trị vào địa chỉ 1000)
    address = 1000  # Địa chỉ cần ghi
    value = 123     # Giá trị cần ghi
    client.write_register(address, value)
    print(f"Ghi giá trị {value} vào địa chỉ {address} thành công.")

    # Đọc dữ liệu từ PLC (Ví dụ: đọc giá trị từ địa chỉ 1000)
    result = client.read_holding_registers(address, 1)
    print(f"Gía trị từ địa chỉ {address}: {result.registers[0]}")

    # Ngắt kết nối
    client.close()
else:
    print("Không thể kết nối đến PLC.")
