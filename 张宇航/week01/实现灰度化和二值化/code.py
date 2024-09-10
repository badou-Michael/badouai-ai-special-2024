#!/usr/bin/env python
__coding__ = "utf-8"
__author__ = "张宇航"

import numpy as np

"""
-------------------------------------------------
   Description :	TODO：
   SourceFile  :	gray_2number
   Author      :	zhangyuhang
   Date	       :	2024-09-06
-------------------------------------------------
"""
import numpy as np
import cv2
import matplotlib.pyplot  as plt
from skimage.color import rgb2gray
# np.set_printoptions(threshold=np.inf)
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h,w], img.dtype)
print(img_gray)
for i in range(h):
    for j in range(w):
        m = img[i,j]       # 取出图像每个点的坐标，因为是cv2读取的，所以是BGR
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)  # 将BGR按给定系数转换为单通道的 grap 灰度

print(m)
print(img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 二值化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
print(img_binary)
print(img_binary.shape)

