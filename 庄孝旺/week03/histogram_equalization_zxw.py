#!/usr/bin/env python
# encoding=gbk
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.title("gray")
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

# 彩色图像直方图均衡化

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

plt.figure()
plt.title("bgr")
plt.hist(bH.ravel(),256, alpha=0.5, color='blue', label='Blue')
plt.hist(gH.ravel(),256, alpha=0.5, color='green', label='Green')
plt.hist(rH.ravel(),256, alpha=0.5,color='red', label='Red')
plt.show()
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", np.hstack([img, result]))

cv2.waitKey(0)

