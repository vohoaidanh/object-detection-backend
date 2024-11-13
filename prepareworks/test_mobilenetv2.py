from facenet_pytorch import MTCNN
from PIL import Image
from time import time

# Khởi tạo MTCNN
mtcnn = MTCNN(keep_all=True)  # Giữ tất cả các khuôn mặt trong ảnh (nếu có nhiều khuôn mặt)

# Đọc ảnh
img = Image.open(r"D:\datasets\widerface\images\train\50_Celebration_Or_Party_houseparty_50_321.jpg")

# Chuyển ảnh thành tensor và phát hiện khuôn mặt
start_time = time()
boxes, probs = mtcnn.detect(img)
print("inference time is :",time()-start_time)
# In các tọa độ của các khuôn mặt
print("Bounding boxes:", boxes)
print("Confidence scores:", probs)

# Hiển thị ảnh với các khung bao quanh khuôn mặt
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots()
ax.imshow(img)

# Vẽ các khung bao quanh các khuôn mặt phát hiện được
if boxes is not None:
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.show()


