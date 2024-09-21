"""
彩色图像的灰度化、二值化

"""
# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

#显示图片
plt.subplot(221)                     #设置图片区域为两行两列的第一个
img = plt.imread("lenna.png")        #读取图片
plt.imshow(img)                      #显示图片
print("------image lenna------")
print(img)                           #打印图片信息
print(img.shape)

#灰度化
img_gray = rgb2gray(img)             #rbg2gray函数把img转化为image_gray
plt.subplot(222)                     #设置图片区域为两行两列的第二个
plt.imshow(img_gray, cmap='gray')    #显示灰度图
print("------image gray------")
print(img_gray)                      #打印灰度图信息
print(img_gray.shape)

#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)    #对灰度图进行二值化处理，对大于等于0.5的取值为1，反之为0
print("------image binary------")
print(img_binary)                    #打印二值图信息
print(img_binary.shape)              #打印二值图的大小

plt.subplot(223)                     #设置图片区域为两行两列的第三个
plt.imshow(img_binary, cmap='gray')  #显示二值图
plt.show()                           #显示画布
