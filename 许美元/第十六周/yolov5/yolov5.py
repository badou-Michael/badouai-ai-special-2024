import cv2
import torch

# 加载YOLOv5模型。第一次需要下载（自动）。
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 读取图片
img = cv2.imread('street.jpg')

# 进行推理
results = model(img)

# 获取检测结果的图像
output_img = cv2.resize(results.render()[0],(512,512))
print(output_img.shape)

# 显示图像
cv2.imshow('YOLOv5', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
