import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

##使用cv2实现灰度化和二值化：

#原图
img = cv2.imread('lenna.png') #不会自动归一化,以bgr格式读取
cv2.imshow('lenna', img)#以bgr格式显示

#灰度图
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_gray = rgb2gray(img) 生成0-1数组， 便可不要19行
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('lenna show gray', img_gray)

#二值图
img_gray = img_gray / 255.0
img_binary = np.where(img_gray >= 0.5, 1.0, 0.0)
cv2.imshow('lenna show binary', img_binary)

##使用plt实现灰度化和二值化

#原图
img = plt.imread('lenna.png')#以rgb格式读取,会自动归一化
plt.subplot(221)
plt.imshow(img)

#灰度图
# img_gray = rgb2gray(img)
h, w = img.shape[:2]
img_gray = np.zeros((h, w), img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = m[0] * 0.3 + m[1] * 0.59 + m[2] * 0.11
plt.subplot(222)
plt.imshow(img_gray, cmap = 'gray')

#二值图
# img_binary = np.where(img_gray >= 0.5, 1, 0)
for i in range(h):
    for j in range(w):
        if img_gray[i, j] >= 0.5:
            img_gray[i, j] = 1
        else:
            img_gray[i, j] = 0
plt.subplot(223)
plt.imshow(img_gray, cmap = 'gray')

plt.show()


