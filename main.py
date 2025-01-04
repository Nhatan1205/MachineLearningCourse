import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu
x = np.linspace(0, 10, 100)  # 100 giá trị từ 0 đến 10
y = np.sin(x)  # Hàm sin(x)

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))  # Tạo canvas với kích thước 8x6 inch
plt.plot(x, y, label="Sine Wave", color="blue", linestyle="-", linewidth=2)

# Thêm tiêu đề và nhãn
plt.title("Simple Line Chart with Matplotlib", fontsize=16)
plt.xlabel("X-axis (Time)", fontsize=12)
plt.ylabel("Y-axis (Amplitude)", fontsize=12)

# Thêm lưới và chú thích
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)

# Hiển thị biểu đồ
plt.show()
