'''
15周作业：实现maskrcnn（简单or手写）
'''

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


