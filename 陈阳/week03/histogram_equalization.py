"""
获取直方图和直方图均衡化
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
获取直方图方式一，使用plt
plt.figure()   新建一个图像
plt.hist(gray.ravel(), bins=256, range=[0, 256]) ：
    gray.ravel()：将灰度图像展平为一维数组。
    bins=256：指定直方图的bin数为256，表示希望将像素值划分为256个bin（即每个像素值有一个bin）
    range=[0, 256]：像素值的范围，从0到256。
"""
image = cv2.imread("../week02/lenna.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.hist(gray_image, 256)
plt.show()

"""
获取直方图方式二，使用cv2
cv2.calcHist([gray_image],[0],None,[256],[0,256]):
    [gray_image]：输入图像，必须放在方括号中，因为函数支持处理多通道图像。
    [0]：表示我们计算的是第0个通道（对于灰度图像，这就是唯一的通道）。
    None：没有掩码，即计算整个图像的直方图。
    [256]：直方图的大小（bin的数量），即将像素值划分为256个bin。
    [0,256]：像素值的范围，从0到256（不包含256）。
"""
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# 创建绘图窗口并设置坐标轴
plt.figure()
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
# 绘制直方图
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

"""
彩色图像直方图，将颜色分为三个通道分别计算，最后合并到一起
"""
channels = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
for (channel, color) in zip(channels, colors):
    flattened_color_hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(flattened_color_hist, color=color)
    plt.show()

"""
直方图均衡化
equalizeHist(src, dst=None)：
    src：图像矩阵(单通道图像)
    dst：默认即可
"""
dst = cv2.equalizeHist(gray_image)
equalize_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
plt.figure()
plt.title("Equalize histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(equalize_hist)
plt.xlim([0, 256])
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray_image, dst]))

"""
彩色直方图均衡化，分成三个通道均衡化再合并
"""
(b, g, r) = cv2.split(image)
b_equal = cv2.equalizeHist(b)
g_equal = cv2.equalizeHist(g)
r_equal = cv2.equalizeHist(r)
result = cv2.merge((b_equal, g_equal, r_equal))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
