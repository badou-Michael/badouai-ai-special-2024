# -*- coding: utf-8 -*-
"""
yzc/20240905
灰度化、二值化
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2
from PIL import Image

#灰度化  三种方法
img = plt.imread("lenna.png")
# 第一种
# img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# 第二种
img_gray = rgb2gray(img)
plt.subplot(221)
plt.imshow(img_gray,cmap='gray')
# plt.show()
# 第三种
img1=cv2.imread("lenna.png")
h,w = img1.shape[:2]
img1_gray= np.zeros([h,w],img1.dtype)
for i in range(h):
    for j in range(w):
        bgr = img1[i,j]
        img1_gray[i,j] = int(bgr[0]*0.11 + bgr[1]*0.59 +bgr[2]*0.3)
# cv2.imshow("yzc imag gray",img1_gray)
# cv2.waitKey()
plt.subplot(224)
plt.imshow(cv2.cvtColor(img1_gray,cv2.COLOR_BGR2RGB))


# 二值化  两种写法
# 第一种
rows,clors = img_gray.shape
# for i in range(rows):
#     for j in range(clors):
#         if (img_gray[i,j] <= 0.5):
#             img_gray[i,j] = 0
#         else:
#             img_gray[i,j] = 1
# 第二种
img_gray=np.where(img_gray <= 0.5, 0,1 )
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')



#展示原图对比
plt.subplot(223)
plt.imshow(img)
plt.show()




