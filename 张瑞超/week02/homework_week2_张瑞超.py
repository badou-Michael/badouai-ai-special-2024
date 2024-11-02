# -*- coding: utf-8 -*-
"""
彩色图像的灰度化、二值化
"""

import numpy as np
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

# 二值化
img_binary = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        if img_gray[i, j]/255 <= 0.5:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255


# 设置行列数和间隙大小
rows = 1  # 图像的行数
cols = 2  # 图像的列数
gap = 50  # 格子之间的间距

# 创建一个空白的“表格”画布，高度为图像高度，宽度为(两个图像宽度 + 中间的gap)
canvas_height = h
canvas_width = w * cols + gap * (cols - 1)
canvas = np.ones((canvas_height, canvas_width), dtype=img.dtype) * 255  # 创建白色背景

# 将第一个图像（灰度图）放在第一个格子里
canvas[0:h, 0:w] = img_gray

# 将第二个图像（二值化图）放在第二个格子里，中间留出空隙
canvas[0:h, w + gap:w * 2 + gap] = img_binary

# 显示组合后的图像
cv2.imshow("Gray, Black & White", canvas)

cv2.waitKey(0)
cv2.destroyAllWindows()

