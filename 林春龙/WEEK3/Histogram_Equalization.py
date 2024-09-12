"""
实现直方图的均衡化
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("lenna.png", 1)
# 获取灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像的直方图，方法一
plt.figure(1)
plt.hist(gray.ravel(), 256)
plt.show()

#直方图均衡化
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure(2)
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
