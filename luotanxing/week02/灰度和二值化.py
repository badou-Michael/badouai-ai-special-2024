from skimage.color import rgb2gray
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


# 灰度化

img = cv.imread('lenna.png')


h, w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)


#原始图
plt.subplot(221)
imgsrc = plt.imread('lenna.png')
plt.imshow(imgsrc)


#灰度图
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')



#灰度图 api
img_gray = rgb2gray(img)
plt.subplot(223)  # 两行一列的第一个子图
plt.imshow(img_gray, cmap='gray')

#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(224)  # 两行一列的第一个子图
plt.imshow(img_binary,cmap='gray')
plt.show()
