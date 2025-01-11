from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

#灰度化 
img = cv2.imread('lenna.png')
h, w= img.shape[:2] #获取图片的high和wide
img_gray = np.zeros((h, w), img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.114 + m[1] * 0.587 + m[2] * 0.299)
cv2.imwrite('lenna_gray.png', img_gray)
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')

#二值化
img_binary = np.where(img_gray/255> 0.5, 1, 0)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
