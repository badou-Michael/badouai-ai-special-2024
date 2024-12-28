import cv2
import numpy as np

# 读取图像
image = cv2.imread('D:\learn\lenna.png')

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用 Canny 函数进行边缘检测
edges = cv2.Canny(gray_image, 50, 150)  

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
