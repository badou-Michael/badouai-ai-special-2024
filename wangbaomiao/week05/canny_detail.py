# -*- coding: utf-8 -*-
# time: 2024/10/26 14:54
# file: canny_detail.py
# author: flame
from pickletools import uint8

import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

from numpy.ma.core import angle

if __name__ == '__main__':
    """
    此代码实现了一个完整的Canny边缘检测算法，包括以下几个步骤：
    1. 图像预处理：将图像像素值从0-1范围转换为0-255范围，并将其转换为灰度图像。
    2. 高斯平滑：使用高斯滤波器对图像进行平滑处理，以减少噪声。
    3. 求梯度：使用Sobel算子计算图像的梯度幅值和方向。
    4. 非极大值抑制：通过比较梯度方向上的邻域像素值，抑制非极大值点。
    5. 双阈值检测：通过设定高低阈值，确定边缘点，并通过连接边缘点来形成连续的边缘。
    6. 绘制结果：显示处理后的图像。
    
    每个步骤的具体实现如下：
    """

    # 将图像像素值从0-1范围转换为0-255范围，以便进行后续处理
    pic_path = "lenna.png"
    img = cv2.imread("lenna.png")
    if pic_path[-4:] == ".png":
        img = img*255
    img = img.mean(axis=-1)
    # 对图像进行灰度化处理，以便简化后续的计算
    img = img.mean(axis=-1)  # 计算每个像素的平均值，将图像转换为灰度图像

    # 1 高斯平滑
    sigma = 0.5  # 高斯核的标准差
    dim = 5  # 高斯核的尺寸
    Gaussian_filter = np.zeros((dim, dim))  # 初始化高斯滤波器
    tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列，用于计算高斯核的位置偏移
    n1 = 1 / (2 * math.pi * sigma ** 2)  # 高斯核的归一化系数
    n2 = -1 / (2 * sigma ** 2)  # 高斯核的指数部分系数
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))  # 计算高斯核的每个元素值
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()  # 归一化高斯核
    dx, dy = img.shape  # 获取图像的高度和宽度
    img_new = np.zeros(img.shape)  # 初始化平滑后的图像
    tmp = dim // 2  # 计算填充的边界大小
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 对图像进行零填充，以便进行卷积操作
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)  # 应用高斯滤波器
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 显示平滑后的图像
    plt.axis('off')  # 关闭坐标轴

    # 2 求梯度、 以下两个是滤波用的sobel矩阵（检测图像中的水平，垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel算子，用于检测水平边缘
    sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel算子，用于检测垂直边缘
    img_tidu_x = np.zeros(img.shape)  # 初始化水平梯度图像
    img_tidu_y = np.zeros([dx, dy])  # 初始化垂直梯度图像
    img_tidu = np.zeros(img.shape)  # 初始化梯度幅值图像
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 对平滑后的图像进行零填充，以便进行卷积操作
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # 计算水平梯度
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # 计算垂直梯度
            img_tidu[i, j] = math.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 计算梯度幅值
    img_tidu_x[img_tidu_x == 0] = 0.0000001  # 避免除零错误
    angle = img_tidu_y / img_tidu_x  # 计算梯度方向
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8), cmap='gray')  # 显示梯度幅值图像
    plt.axis('off')  # 关闭坐标轴

    # 3 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)  # 初始化非极大值抑制后的图像
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 标记当前像素是否为局部最大值
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 获取当前像素的3x3邻域
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]  # 计算左上角的插值
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]  # 计算右下角的插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False  # 当前像素不是局部最大值
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]  # 计算右上角的插值
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]  # 计算左下角的插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False  # 当前像素不是局部最大值
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]  # 计算右上方的插值
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]  # 计算左下方的插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False  # 当前像素不是局部最大值
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]  # 计算左上方的插值
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]  # 计算右下方的插值
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False  # 当前像素不是局部最大值
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]  # 保留局部最大值
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 显示非极大值抑制后的图像
    plt.axis('off')  # 关闭坐标轴

    # 4 双阈值检测 连接边缘 遍历所有一定是边的点，查看8领域是否存在有可能是边的点 进栈
    lower_boundary = img_tidu.mean() * 0.5  # 低阈值，设置为梯度幅值均值的0.5倍
    high_boundary = lower_boundary * 3  # 高阈值，设置为低阈值的3倍
    zhan = []  # 初始化栈，用于存储边缘点
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255  # 标记为边缘
                zhan.append([i, j])  # 将边缘点加入栈
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0  # 标记为非边缘

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]  # 获取当前像素的3x3邻域
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0  # 将非边缘点标记为0

    # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')  # 显示最终的边缘检测结果
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像
