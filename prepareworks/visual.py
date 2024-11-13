import cv2
import matplotlib.pyplot as plt

def draw_bounding_boxes(image_path, label_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    
    # Đọc file label
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
            
            # Chuyển tọa độ YOLO về tọa độ pixel
            x1 = int((x_center - bbox_width / 2) * width)
            y1 = int((y_center - bbox_height / 2) * height)
            x2 = int((x_center + bbox_width / 2) * width)
            y2 = int((y_center + bbox_height / 2) * height)
            
            # Vẽ bounding box lên ảnh
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Hiển thị ảnh với bounding boxes
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Ví dụ sử dụng
    draw_bounding_boxes(r"D:\datasets\widerface\images\train\0_Parade_marchingband_1_849.jpg", r"D:\datasets\widerface\labels\train\0_Parade_marchingband_1_849.txt")
