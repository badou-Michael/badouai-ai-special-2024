#直方图均衡化

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread("lenna.png")
# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 直方图均衡化
equalized = cv2.equalizeHist(gray)
# 计算均衡化后的图像的直方图
hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
plt.figure(figsize=(8, 6))
plt.plot(hist_equalized, color='gray')
plt.title('Equalized Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
plt.grid(axis='y', alpha=0.75)

# 保存图形到文件
plt.savefig('lenna03_3.png')  # 保存为PNG文件
