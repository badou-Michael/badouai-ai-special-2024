# -*- coding: utf-8 -*-
# author: 王博然
import cv2
import numpy as np

img = cv2.imread("./lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h,w], img.dtype)
img_binary = np.zeros([h,w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i][j]
        img_gray[i][j] = (m[0]*28 + m[1]*151 + m[2]*76) >> 8
        if img_gray[i][j] > 122:
            img_binary[i][j] = 255
        else:
            img_binary[i][j] = 0

# 打印灰度化
cv2.imshow("image show gray", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 打印二值化
cv2.imshow("image show binary", img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()