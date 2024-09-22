import cv2
import numpy as np
from matplotlib import pyplot as plt


# 灰度图像直方图
# 获取灰度图像
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image_gray" ,img_gray)

# 灰度化图像的直方图，方法一
# 创建一个新窗口
plt.figure()
# 使用hist函数绘制灰度图像的直方图
# img_gray.ravel()将二维的灰度图像数组转换为一维数组，以便绘制直方图。 256是直方图的bin数量，表示灰度级别从0到255）
# ravel()方法将这个二维数组“展平”为一维数组，这样每个像素的灰度值都变成了一个单独的元素。这是绘制直方图所必需的，因为直方图需要一维的数据序列。
plt.hist(img_gray.ravel(), 256)
# 显示图形窗口
plt.show()
