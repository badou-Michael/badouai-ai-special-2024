#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/10/16 20:12
@Author  : Mr.Long
"""
import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

# from util_tools.util_tools import UtilTools

class HomeworkW5(object):
    """Canny Edge Detection：Canny边缘检测
    1. 对图像进行灰度化：用取均值的方法image.mean(axis=-1) ，plt.read(image)的结果是0-1，需要*255再均值化
    2. 对图像进行高斯滤波（高斯平滑）：
    3. 检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）。
    4. 对梯度幅值进行非极大值抑制
    5. 用双阈值算法检测和连接边缘
    """
    def canny_detail_step_one(self, image_array):
        """均值灰度化"""
        image_gray = image_array.mean(axis=-1)
        return image_gray

    def canny_detail_step_two(self, image_gray, sigma, dim):
        """高斯平滑:
        a.高斯核参数，标准差sigma值可调，高斯核尺寸dim=5
        b.存储高斯核np.zero([dim, dim])
        c.根据dim生成一个整数序列，长度是5
        d.根据高斯公式计算高斯核G(x,y)= n1*math.exp(n2*(x**2+y**2)),n1=1/2*math.pi*sigma**2,n2=-1/2*sigma**2
        e.对高斯函数进行归一化处理
        f.定义一个高宽和原图一样的零矩阵，用np.pad对灰度后的图片进行边缘填补
        g.将滤波后的数据赋值给零矩阵，即是平滑后的图片"""
        gaussian_filter = np.zeros([dim, dim])
        temp_list = [i - dim//2 for i in range(dim)]
        n1, n2 = 1 / 2 * math.pi * sigma**2, -1 / 2 * sigma**2
        for i in range(dim):
            for j in range(dim):
                gaussian_filter[i, j] = n1 * math.exp(n2 * (temp_list[i]**2 + temp_list[j]**2))
        gaussian_filter = gaussian_filter / gaussian_filter.sum()
        dh, dw = image_gray.shape
        new_image = np.zeros(image_gray.shape)
        tmp = dim // 2
        img_pad = np.pad(image_gray, ((tmp, tmp), (tmp, tmp)), 'constant')
        for i in range(dh):
            for j in range(dw):
                new_image[i, j] = np.sum(img_pad[i:i+dim, j:j+dim] * gaussian_filter)
        return new_image

    def canny_detail_step_three(self, new_image):
        """求梯度：边缘
        a.创建sobel的横轴、纵轴卷积核
        b.以原图的高宽创建三个零矩阵，横向、纵向、以及结果矩阵
        c.进行边缘填充，当前是加1圈
        d.通过对横轴、纵轴进行单独卷积得到卷积后的结果，也就是横轴、纵轴的向量，计算出最终向量
        f.对横向卷积结果为0的点赋值，变成非0，方便计算角度"""
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        dh, dw = new_image.shape
        image_edge_x, image_edge_y, image_edge = (np.zeros(new_image.shape), np.zeros(new_image.shape),
                                                  np.zeros(new_image.shape))
        img_pad = np.pad(new_image, ((1, 1), (1, 1)), 'constant')
        for i in range(dh):
            for j in range(dw):
                image_edge_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
                image_edge_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
                image_edge[i, j] = np.sqrt(image_edge_x[i, j] ** 2 + image_edge_y[i, j] ** 2)
        image_edge_x[image_edge_x == 0] = 0.00000001
        tangent = image_edge_y / image_edge_x
        return tangent, image_edge

    def canny_detail_step_four(self, tangent, image_edge):
        """非极大值抑制"""
        suppression = np.zeros(image_edge.shape)
        dh, dw = image_edge.shape
        for i in range(1, dh - 1):
            for j in range(1, dw - 1):
                flag = True
                temp = image_edge[i - 1:i + 2, j - 1:j + 2]
                if tangent[i, j] <= -1:
                    num1 = (temp[0, 1] - temp[0, 0]) / tangent[i, j] + temp[0, 1]
                    num2 = (temp[2, 1] - temp[2, 2]) / tangent[i, j] + temp[2, 2]
                    if not (image_edge[i,j] > num1 and image_edge[i,j] > num2):
                        flag = False
                elif tangent[i, j] >= 1:
                    num1 = (temp[0, 2] - temp[0, 1]) / tangent[i, j] + temp[0, 1]
                    num2 = (temp[2, 0] - temp[2, 1]) / tangent[i, j] + temp[2, 1]
                    if not (image_edge[i, j] > num1 and image_edge[i, j] > num2):
                        flag = False
                elif tangent[i, j] > 0:
                    num1 = (temp[1, 2] - temp[0, 2]) / tangent[i, j] + temp[0, 2]
                    num2 = (temp[1, 0] - temp[2, 0]) / tangent[i, j] + temp[2, 0]
                    if not (image_edge[i, j] > num1 and image_edge[i, j] > num2):
                        flag = False
                elif tangent[i, j] < 0:
                    num1 = (temp[1, 0] - temp[0, 0]) / tangent[i, j] + temp[1, 0]
                    num2 = (temp[1, 2] - temp[2, 2]) / tangent[i, j] + temp[1, 2]
                    if not (image_edge[i, j] > num1 and image_edge[i, j] > num2):
                        flag = False
                if flag:
                    suppression[i, j] = image_edge[i, j]
        return suppression
    
    def canny_detail_step_five(self, suppression, lower_boundary, high_boundary):
        """双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
        """
        stack = []
        for i in range(1, suppression.shape[0] - 1):
            for j in range(1, suppression.shape[1] - 1):
                if suppression[i, j] >= high_boundary:
                    suppression[i, j] = 255
                    stack.append([i, j])
                elif suppression[i, j] <= lower_boundary:
                    suppression[i, j] = 0
        while not len(stack) == 0:
            temp_1, temp_2 = stack.pop()
            a = suppression[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
                suppression[temp_1 - 1, temp_2 - 1] = 255
                stack.append([temp_1 - 1, temp_2 - 1])
            if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
                suppression[temp_1 - 1, temp_2] = 255
                stack.append([temp_1 - 1, temp_2])
            if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
                suppression[temp_1 - 1, temp_2 + 1] = 255
                stack.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
                suppression[temp_1, temp_2 - 1] = 255
                stack.append([temp_1, temp_2 - 1])
            if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
                suppression[temp_1, temp_2 + 1] = 255
                stack.append([temp_1, temp_2 + 1])
            if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
                suppression[temp_1 + 1, temp_2 - 1] = 255
                stack.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
                suppression[temp_1 + 1, temp_2] = 255
                stack.append([temp_1 + 1, temp_2])
            if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
                suppression[temp_1 + 1, temp_2 + 1] = 255
                stack.append([temp_1 + 1, temp_2 + 1])
        for i in range(suppression.shape[0]):
            for j in range(suppression.shape[1]):
                if suppression[i, j] != 0 and suppression[i, j] != 255:
                    suppression[i, j] = 0
        return suppression
                    



if __name__ == '__main__':
    homework_w5 = HomeworkW5()
    # image_path = UtilTools.get_file_path('lenna.png')
    image_path = 'lenna.png'
    image_array = cv2.imread(image_path)
    image_gray = homework_w5.canny_detail_step_one(image_array)
    gaussian_filter = homework_w5.canny_detail_step_two(image_gray, 0.5, 5)
    tangent, image_edge = homework_w5.canny_detail_step_three(gaussian_filter)
    old_suppression = homework_w5.canny_detail_step_four(tangent, image_edge)
    lower_boundary = image_edge.mean() * 0.5
    high_boundary = lower_boundary * 3
    suppression = homework_w5.canny_detail_step_five(old_suppression, lower_boundary, high_boundary)
    plt.imshow(suppression.astype(np.uint8), cmap='gray')
    plt.axis('off') 
    plt.show()
