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
img = cv2.imread("../../request/task2/lenna.png", 1)
# bgr转rgb
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)
# 灰度图像直方图均衡化，直接调用
dst = cv2.equalizeHist(gray)
# 计算均衡化后的直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
"""
plt.figure()用于创建一个新的图形对象(Figure)，它是绘图的最顶层容器。
可以使用该对象进行图形的设置和操作，例如设置图形的大小、标题等。
plt.hist()函数的主要作用是绘制‌直方图，用于展示数据的分布情况, bins参数用于指定数据分成的区间数量
"""
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)


