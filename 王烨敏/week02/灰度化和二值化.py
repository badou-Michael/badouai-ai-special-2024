# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

plt.subplot(221)
img=plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna---")
print("ing")

#灰度化
img_gray=rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray---")
print(img_gray)

#二值化
img_binary=np.where(img_gray >= 0.5,1,0)
print("---img_binary---")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()
