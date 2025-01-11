'''
1.实现yolov3
2.实现mtcnn
'''

#1、实现yolov3
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 加载预训练的 YOLOv3 模型
model = YOLO('yolov3.pt')  # 确保你已下载 yolov3.pt 权重文件

# 读取本地图片
image_path = 'street.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换 BGR 为 RGB 以便 Matplotlib 显示

# 进行目标检测
results = model(img)

# 显示检测结果
results[0].show()  # 弹窗显示检测结果

# 可视化检测结果并绘制
plt.imshow(results[0].plot())  # Matplotlib 显示检测到的边界框
plt.axis('off')
plt.show()

#2、实现mtcnn
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# 初始化 MTCNN 模型
mtcnn = MTCNN(keep_all=True)  # keep_all=True 保证检测多张人脸

# 读取图片
image_path = 'street.jpg'  # 替换为你的图片路径
img = Image.open(image_path)

# 进行人脸检测
boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

# 绘制人脸边框和关键点
draw = ImageDraw.Draw(img)
if boxes is not None:
    for box, landmark in zip(boxes, landmarks):
        draw.rectangle(box.tolist(), outline=(255, 0, 0), width=3)  # 绘制人脸框
        for point in landmark:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(0, 255, 0))  # 绘制关键点

# 显示结果
plt.imshow(img)
plt.axis('off')
plt.show()
