# -*- coding: utf-8 -*-
"""
@author: YvanHall

图像灰度化和二值化示例
使用cv2实现
"""

import numpy as np
import cv2

# 一、灰度化

# 读取图片
img = cv2.imread('lenna.png')
# 获取宽高
high,width = img.shape[:2]
# 创建同等大小单通道图片
img_gray = np.zeros([high,width], img.dtype)
# 取出BGR坐标
for i in range(high):
    for j in range(width):
        m = img[i,j]
        # BGR转化为gray
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
# 生成灰度图
cv2.imwrite("gray_lenna.png", img_gray)

# 二值化
ret, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# 生成二值化图
cv2.imwrite("binary_lenna.png", thresh)
