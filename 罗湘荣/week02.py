
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#灰度化
photo=cv2.imread("ho.jpg")#读取图片
hight,wight=photo.shape[:2]#获取高和宽
photo_gray=np.zeros([hight,wight],photo.dtype) #创建和原图一样的大小的灰度图片
for i in range(hight):
    for j in range(wight):
        m=photo[i,j]
        photo_gray[i,j]= int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
cv2.imshow("灰度化后的图片",photo_gray)


#二值化（在灰度图的基础上二值化）
photo_binary=np.where(photo_gray >= 128, 255, 0).astype(np.uint8)
cv2.imshow("二值化图像",photo_binary)


#三图对比
#原图
plt.subplot(221)
photo = plt.imread("ho.jpg")
plt.imshow(photo)
#灰度
plt.subplot(222)
photo_gray = rgb2gray(photo)
plt.imshow(photo_gray,cmap='gray')
#二值化
plt.subplot(223)
photo_binary=np.where(photo_gray>=0.5 ,1 ,0)
plt.imshow(photo_binary,cmap='gray')
plt.show()

