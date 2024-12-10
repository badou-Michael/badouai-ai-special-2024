#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
# [dst]:
# 这是一个包含图像数据的列表。在这个上下文中，dst 是一个图像（通常是单通道或多通道图像），我们想要计算这个图像的直方图。
# [0]:
# 这个参数指定要计算直方图的通道索引。对于灰度图像，通常使用 0 来表示第一通道（也是唯一通道）。如果是彩色图像，可以使用 0, 1, 2 来分别表示蓝色、绿色和红色通道。
# None:
# 这个参数是用于掩模的。在这里，None 表示我们不使用任何掩模，也就是说，我们将计算整个图像的直方图。
# [256]:
# 这个参数指定直方图的大小。在这里，256 表示我们想要创建一个包含 256 个 bin 的直方图，通常用于表示像素值从 0 到 255 的分布。
# [0, 256]:
# 这个参数定义了每个 bin 的范围。在这里，范围是从 0 到 256，这意味着每个 bin 将对应于一个像素值（0-255）。

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)


'''
# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''
