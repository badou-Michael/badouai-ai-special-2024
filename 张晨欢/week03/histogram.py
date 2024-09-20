import cv2
import numpy as np
from matplotlib import pyplot as plt
# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像的直方图，方法一
plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()
