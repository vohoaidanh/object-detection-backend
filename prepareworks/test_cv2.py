import cv2
import matplotlib.pyplot as plt
# Đọc ảnh từ đường dẫn
img = cv2.imread(r"D:\datasets\real_gen_dataset\val\1_fake\0b745c91-8ffe-4cf7-a815-ebeb1b6c1d5c.jpg")

# Kiểm tra nếu ảnh đã được tải thành công
if img is not None:
    # Hiển thị ảnh trong một cửa sổ
    plt.imshow(img)
    plt.show()
