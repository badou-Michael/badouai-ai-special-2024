# -*- coding: utf-8 -*-
"""
彩色图像灰度化、二值化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# 灰度化
img = cv2.imread("lenna.png")
height,width = img.shape[:2]
img_gray = np.zeros((height, width), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 +m[2]*0.3)

print(m,'\n')
print(img,'\n')
print("image show gray:",img_gray)
# plt.subplot(221)
cv2.imshow('Original Color Image', img)
# plt.subplot(222)
cv2.imshow('Generated Gray Image', img_gray)
# cv2.waitKey(0)

#二值化
rows, cols = img_gray.shape
img_binary = np.zeros((rows, cols), dtype=np.uint8)
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j]<=127:
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255
print("Binary Image:", img_binary)
cv2.imshow('Binary Image', img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()


#二值化
# rows, cols = img_gray.shape
# img_binary = np.zeros((rows, cols), dtype=np.uint8)
# for i in range(rows):
#     for j in range(cols):
#         if img_gray[i, j] <= 127:
#             img_binary[i, j] = 0
#         else:
#             img_binary[i, j] = 255
#
# print("Binary Image:", img_binary)
# cv2.imshow('Binary Image', img_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # 灰度化
# img_gray = rgb2gray(img)
# # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # img_gray = img
# plt.subplot(222)
# plt.imshow(img_gray, cmap='gray')
# print("---image gray----")
# print(img_gray)
#
# # 二值化
# # rows, cols = img_gray.shape
# # for i in range(rows):
# #     for j in range(cols):
# #         if (img_gray[i, j] <= 0.5):
# #             img_gray[i, j] = 0
# #         else:
# #             img_gray[i, j] = 1
#

# img_binary = np.where(img_gray >= 0.5, 1, 0)
# print("-----imge_binary------")
# print(img_binary)
# print(img_binary.shape)
#
# plt.subplot(223)
# plt.imshow(img_binary, cmap='gray')
# plt.show()
