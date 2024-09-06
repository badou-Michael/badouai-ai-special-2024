"""
lenna图像的灰度化
"""
#导入库
import skimage.color
import cv2
import numpy as np
import matplotlib.pyplot as plt

#灰度化
from numpy import ndarray

img = cv2.imread("lenna.png")  #读取图片，转换为NumPy数组
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
img_gray2 = np.zeros([h,w],img.dtype)
#方法一
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

#方法二
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('gray',img_gray)
#cv2.imshow('gray2',img_gray2)
cv2.waitKey(0)

#二值化
row,col = img_gray.shape
img_binary = np.zeros([h,w],img_gray.dtype)
threshold = 110
for i in range(row):
    for j in range(col):
        if (img_gray[i, j] <= threshold) :
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 255

#img_binary = np.where(img_gray >= 0.5, 1, 0)
'''plt.subplot(111)
plt.imshow(img_binary,cmap ='gray')
plt.show()'''
cv2.imshow('binary',img_binary)
cv2.waitKey(0)
