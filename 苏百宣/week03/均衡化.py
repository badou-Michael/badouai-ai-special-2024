import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并将其转换为灰度图
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# 显示原始图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.colorbar()

# 使用OpenCV的equalizeHist函数进行直方图均衡化
equalized_img = cv2.equalizeHist(img)

# 显示直方图均衡化后的图像
plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_img, cmap='gray')
plt.colorbar()

plt.show()

# 计算直方图并显示
plt.figure(figsize=(10, 5))

# 原始图像直方图
plt.subplot(1, 2, 1)
plt.title("Original Image Histogram")
plt.hist(img.ravel(), 256, [0, 256])
plt.xlim([0, 256])

# 均衡化后图像的直方图
plt.subplot(1, 2, 2)
plt.title("Equalized Image Histogram")
plt.hist(equalized_img.ravel(), 256, [0, 256])
plt.xlim([0, 256])

plt.show()
