# 使用yolo5模型推理

import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

img = cv2.imread('street.jpg')
results = model(img)

output_img = cv2.resize(results.render()[0], (512, 512))
print(output_img.shape)

cv2.imshow('YOLOv5', output_img)
cv2.waitKey(0)
