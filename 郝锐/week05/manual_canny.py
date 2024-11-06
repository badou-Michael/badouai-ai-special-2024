#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/10/18 14:29
# @Author: Gift
# @File  : manual_canny_edge_extract.py 
# @IDE   : PyCharm
import cv2
import math

import numpy as np

from gauss_nosie import img_gauss

"""
1.灰度化,降低计算难度
2.高斯滤波,平滑图像,减少噪声
3.计算梯度幅值和方向
4.非极大值抑制,细化边缘
5.双阈值检测,确定强边缘和弱边缘
"""
#1 读取灰度图
#img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("lenna.png",0)
#2 高斯滤波
kernel_size = 5
#生成一个5*5数组，为高斯核做准备
gaussian_kernel = [[0] * kernel_size for _ in range(kernel_size)] #高斯核
print(gaussian_kernel)
center = kernel_size // 2 #高斯核中心
print(center)
sigma = 1.5 #标准差-可调
sum_weight = 0
#根据高斯函数计算每个位置的权重值
for i in range(kernel_size):
    for j in range(kernel_size):
        x, y = i - center, j - center
        weight = (1 / (2 * math.pi * sigma ** 2)) * math.e**(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        gaussian_kernel[i][j] = weight
        sum_weight += weight
print(gaussian_kernel)
#归一化权重值
for i in range(kernel_size):
    for j in range(kernel_size):
        gaussian_kernel[i][j] /= sum_weight
print(gaussian_kernel)
#卷积操作
#定义一个和原图像大小一样的空白图像来存储高斯滤波后的图像
gaussian_img = np.zeros(img.shape, dtype=np.uint8)
#添加padding
padding = kernel_size // 2
img_pad = np.pad(img, (padding, padding), mode='constant')
#卷积操作
x,y = img.shape
for i in range(x):
    for j in range(y):
        gaussian_img[i,j] = int(np.sum(img_pad[i:i+kernel_size, j:j+kernel_size] * gaussian_kernel))
cv2.imshow("gaussian_img", gaussian_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#3 计算梯度幅值和方向
sobel_x = cv2.Sobel(gaussian_img, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gaussian_img, cv2.CV_64F, 0, 1, ksize=5)
print("sobel_x")
print(sobel_x)
print("sobel_y")
print(sobel_y)
magnitude = cv2.magnitude(sobel_x, sobel_y)
print("magnitude")
print(magnitude)
orientation = cv2.phase(sobel_x, sobel_y, angleInDegrees=True)
print("orientation")
print(orientation)
#4 非极大值抑制
img_suppress = np.copy(magnitude)
for i in range(1, magnitude.shape[0] - 1):
    for j in range(1, magnitude.shape[1] - 1):
        angle = orientation[i, j]
        if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
            if magnitude[i, j] < magnitude[i, j + 1] or magnitude[i, j] < magnitude[i, j - 1]:
                img_suppress[i, j] = 0
        elif (22.5 <= angle < 67.5):
            if magnitude[i, j] < magnitude[i + 1, j + 1] or magnitude[i, j] < magnitude[i - 1, j - 1]:
                img_suppress[i, j] = 0
        elif (67.5 <= angle < 112.5):
            if magnitude[i, j] < magnitude[i + 1, j] or magnitude[i, j] < magnitude[i - 1, j]:
                img_suppress[i, j] = 0
        else:
            if magnitude[i, j] < magnitude[i + 1, j - 1] or magnitude[i, j] < magnitude[i - 1, j + 1]:
                img_suppress[i, j] = 0
print("img_suppress")
print(img_suppress)
#5 双阈值检测
low_threshold = magnitude.mean()*0.99 #低阈值,一般取高斯滤波后图像梯度的平均值的一半
high_threshold = low_threshold*3 #高阈值,一般取低阈值的3倍
edge_img = magnitude.copy()
for i in range(magnitude.shape[0]-1):
    for j in range(magnitude.shape[1]-1):
        if edge_img[i, j] > high_threshold:
            edge_img[i, j] = 255 #强边缘,保留
        elif low_threshold <= edge_img[i, j] <= high_threshold:
            #若边缘，计算其周围的八个点是否有强边缘，有则保留，没有则抑制
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if edge_img[i + x, j + y] > high_threshold:
                        edge_img[i, j] = 255
                        break
        else:
            edge_img[i, j] = 0 #弱边缘,抑制
cv2.imshow("edge_img", edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



