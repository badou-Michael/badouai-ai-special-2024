# -*- coding: utf-8 -*-
"""
@author: SJLong

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化方法一
'''获取图片的high和wide, 取出第1列和第2列的值，用前2列的值创建一张和当前图片大小一样的单通道图片，
   取出当前high和wide中的BGR坐标，用浮点数公式将BGR坐标转化为gray坐标并赋值给新图像'''
def ImageGrayOne():
    img = cv2.imread("../../data/lenna.png")
    high, wide = img.shape[:2]
    img_gray = np.zeros([high, wide], img.dtype)
    for i in range(high):
        for j in range(wide):
            m = img[i,j]
            img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    # 用plt显示图片
    # plt.subplot(221)
    # plt.imshow(img_gray, cmap='gray')
    # plt.show()
    # 用cv2显示图片
    cv2.imshow("image show gray", img_gray)
    cv2.waitKey()



# 灰度化方法二
'''读取图片，直接使用函数进行灰度，用plt显示'''
def ImageGrayTwo():
    img = cv2.imread("../../data/lenna.png")
    img_gray = rgb2gray(img)
    plt.subplot(222)
    plt.imshow(img_gray, cmap='gray')
    plt.show()




# 二值化方法一，通常是指单通道且只有0和255两个值
'''读取图片，灰度化后再使numpy的where进行二值化，用plt显示'''
def ImageBinaryOne():
    img = cv2.imread("../../data/lenna.png")
    img_gray = rgb2gray(img)
    img_binary = np.where(img_gray >= 0.5, 1, 0)
    plt.subplot(223)
    plt.imshow(img_binary, cmap='gray')
    plt.show()


# 二值化方法二，通常是指单通道且只有0和255两个值
'''读取图片，设置常量进行判断,将值除255转化成0和1'''
def ImageBinaryTwo():
    img = cv2.imread("../../data/lenna.png")
    img_gray = rgb2gray(img)
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if img_gray[i, j] <= 0.5:
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 1
    plt.subplot(224)
    plt.imshow(img_gray, cmap='gray')
    plt.show()


ImageGrayOne()
ImageGrayTwo()
ImageBinaryOne()
ImageBinaryTwo()

