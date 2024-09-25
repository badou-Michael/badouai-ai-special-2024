# -*- coding: utf-8 -*-
"""
@author: 赵冬
@since: 2024年9月25日
彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

"""
使用for循环方式
"""
def fn1():
    img = cv2.imread("lenna.png")
    h,w,_  = img.shape
    img_gray = np.zeros([h,w], dtype=img.dtype)

    for i in range(h):
        for j in range(w):
            m = img[i,j]
            img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.37)

    cv2.imwrite("lenna_gray1.png", img_gray)

    img_binary = np.zeros([h,w], dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            m = img_gray[i,j]
            img_binary[i,j] = 0 if m < 128 else 255
    cv2.imwrite("img_binary1.png", img_binary)

"""
使用np
"""
def fn2():
    img = cv2.imread("lenna.png")
    img_gray = rgb2gray(img)
    cv2.imwrite("lenna_gray2.png", (img_gray * 255).astype(int))

    img_binary = np.where(img_gray <= 0.5, 0, 255)
    cv2.imwrite("img_binary2.png", img_binary)

if __name__ == '__main__':
    fn1()
    fn2()
