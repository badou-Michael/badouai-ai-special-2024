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
img = cv2.imread("lenna.png", 1)   ##1表示以彩色样式读取
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
"""
cv2.calcHist()：这是 OpenCV 中计算图像直方图的函数。
[dst]：图像的输入。在这里，dst 是图像的矩阵（通常是一个灰度图像）。这个参数是一个列表形式，即使是单张图像，也需要写成 [dst]。
[0]：表示计算第 0 个通道的直方图。对于灰度图像，只有一个通道，所以这里是 0；对于彩色图像，BGR 有 3 个通道，可以分别计算各个通道的直方图。
None：表示不使用掩码，即对整个图像计算直方图。
[256]：表示将像素值分为 256 个区间（通常的灰度值范围是 0 到 255）。对于彩色图像，同样适用此设置。
[0,256]：表示像素值的范围是 0 到 255，定义了直方图的范围。
"""



plt.figure()
plt.hist(dst.ravel(), 256)
plt.show() ##显示灰度直方图

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)  ##显示均衡化后的图片


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
