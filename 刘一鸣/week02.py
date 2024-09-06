"""
@author: Yiming
@date:20240906
WEEK2作业：实现灰度化和二值化

注意plt和cv2读取照片处理的RGB顺序不一样导致输出也不一样
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

img=cv2.imread("lenna.png")
img2=plt.imread("lenna.png")
h,w=img.shape[:2]
img_gray=np.zeros([h,w],img.dtype)

for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)  #注意这里是BGR而不是RGB

img_binary=np.where(img_gray>=123,1,0)

plt.subplot(221),plt.imshow(img)
plt.subplot(222),plt.imshow(img_gray,cmap='gray')
plt.subplot(223),plt.imshow(img_binary,cmap='gray')
plt.subplot(224),plt.imshow(img2)
plt.show()

