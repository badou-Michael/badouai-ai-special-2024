import cv2
import torch

#加载yolov5模型
model=torch.hub.load('ultralytics/yolov5','yolov5s')

#读取图片
img=cv2.imread('street.jpg')

#推理
results=model(img)

#显示图像
cv2.imshow('yolov5',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
