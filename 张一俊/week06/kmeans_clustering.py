import cv2
import numpy as np

# 读取图像并转换为数据
image = cv2.imread('lenna.png')
Z = image.reshape((-1, 3))  # 将图像转换为一个二维数组
Z = np.float32(Z)  # 转换为浮点数

# 定义 K-means 的停止条件和参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4  # 聚类数量

# 应用 K-means 聚类
_, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 转换中心点为 uint8，并重建聚类后的图像
centers = np.uint8(centers)
result = centers[labels.flatten()]
result = result.reshape(image.shape)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('K-means Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
