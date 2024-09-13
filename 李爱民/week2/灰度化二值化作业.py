# -*- coding: utf-8 -*-
"""
@author: 李爱民

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# 图像获取
image = cv2.imread("D:\lenna.png") #获取图片


image1 = cv2.bitwise_not(image) #函数cv2.bitwise_not可以实现像素点各通道值取反

cv2.imshow("image", image)

cv2.imshow("image1", image1)

# print(image)
 #print(image1)

#灰度化


imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #彩色转灰度

cv2.imshow("imgray", imgray)

thresh=120

ret,binary = cv2.threshold(imgray,thresh,255,cv2.THRESH_BINARY) #输入灰度图，输出二值图

print("-----binary------")
print(binary)
print(binary.shape)


binary1 = cv2.bitwise_not(binary) #取反

plt.imshow(binary1, cmap='gray')
plt.show()
