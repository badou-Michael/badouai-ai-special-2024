import cv2
import numpy as np

# 读取图像
image = cv2.imread('River.webp')

# 定义原始图像中矩形的四个顶点坐标（这里假设已经知道这些坐标）
pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

# 定义目标矩形的四个顶点坐标（这里将其校正为一个正面的矩形）
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)

# 应用透视变换
result = cv2.warpPerspective(image, M, (300, 300))

# 显示原始图像和变换后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Transformed Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()