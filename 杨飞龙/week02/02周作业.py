# -*- coding: utf-8 -*-
"""
@author: 杨飞龙

彩色图像的灰度化、二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#灰度化（手动）
# img = cv2.imread ( 'lenna.png' )
# h,w = img.shape[0:2]
# img_gray = np.zeros([h,w],img.dtype)
# for i in range(h):
#     for j in range(w):
#         m = img[i,j]
#         img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
# cv2.imshow('gray',img_gray)
# cv2.waitKey(0)

# 灰度化（调接口）
img = plt.imread("lenna.png")
img_gray = rgb2gray(img)
plt.subplot(121)
plt.imshow(img_gray,cmap="gray")

print("-----image-----")
print(img_gray)

# 二值化 (手动cv2)
# g,k = img_gray.shape[:2]
# for i in range(g):
#     for j in range(k):
#         if (img_gray[i, j] <= 125):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 255
#
# cv2.imshow('binary',img_gray)
# cv2.waitKey(0)

#二值化（手动plt ）
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
# cv2.imshow('binary',img_gray)
# cv2.waitKey(0)

#二值化（调接口）
img_binary = np.where(img_gray >= 0.5, 1, 0)

plt.subplot(122)
plt.imshow(img_binary,cmap="gray")
plt.show()




