import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
calcHist 计算图像直方图
函数模型：calHist([image], [channels], [mask], [histSize], [ranges], [hist = None], [accumulate = None])
[images]: 想要操作的图像类表
[channels]:[0],表示对图像第一个通道进行直方图计算
[mask]：掩码。一般为[None]，表示没有掩码，考虑所有像素
[histSize]：表示直方图的bin数量，直方图大小，一般等于灰度等级。
当我们说直方图的bin数量是256时，意味着直方图将有256个区间，每个区间对应一个灰度级。这样，直方图可以精确地统计每个灰度级在图像中出现的次数
[ranges]：像素值范围
"""

# 灰度化直方图
img = cv2.imread("lenna.png")
img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img_gary], [0], None, [256], [0, 256])  # 计算直方图
plt.figure()  # 新建一个新的图形窗口
plt.title("Grayscale Histogram")  # 给这个图形设置标题
plt.xlabel("Bins")  # x轴标签
plt.ylabel("# of Pixels")  # y轴标签
plt.plot(hist)  # 绘制直方图
plt.xlim([0, 256])  # 设置x坐标轴范围
plt.show()