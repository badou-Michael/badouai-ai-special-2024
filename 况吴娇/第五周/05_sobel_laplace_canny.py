#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png",1)  #1表示以彩色模式读取图像。
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#Sobel 算子是一种用于图像处理的离散微分算子，它结合了高斯平滑和微分求导，用于计算图像亮度的空间梯度，从而检测图像中的边缘。
'''
Sobel算子
Sobel算子函数原型如下：
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
前四个是必须的参数：
第一个参数是需要处理的图像；
第二个参数是图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度；
dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2。
其后是可选的参数：
dst是目标图像；
ksize是Sobel算子的大小，必须为1、3、5、7。
scale是缩放导数的比例常数，默认情况下没有伸缩系数；
delta是一个可选的增量，将会加到最终的dst中，同样，默认情况下没有额外的值加到dst中；
borderType是判断图像边界的模式。这个参数默认值为cv2.BORDER_DEFAULT。
'''
# 1, 0和0, 1：分别表示在x和y方向上的求导阶数。
# ksize=3：Sobel算子的大小，这里使用3x3的核。
img_sobel_x=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)  #cv2.CV_64F：目标图像的深度，使用64位浮点数。
img_sobel_y=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)


# Laplace 算子
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)  ##laplace算子是另一种边缘检测方法，它通过计算图像的二阶导数来检测边缘。

# Canny 算子
img_canny = cv2.Canny(img_gray, 100, 150)  #Canny算子是一种非常流行的边缘检测算法，它通过多阶段处理来检测边缘。

#plt.imshow(img_gray, "gray")  # 使用灰度颜色映射显示图像
plt.subplot(231),plt.imshow(img_gray, "gray"),plt.title('original')
plt.subplot(232),plt.imshow(img_sobel_x, "gray"),plt.title('Sobel_x')
plt.subplot(233),plt.imshow(img_sobel_y, "gray"),plt.title('Sobel_y')
plt.subplot(234),plt.imshow(img_laplace, "gray"),plt.title('img_laplace')
plt.subplot(235),plt.imshow(img_canny, "gray"),plt.title('img_canny')


plt.show()


