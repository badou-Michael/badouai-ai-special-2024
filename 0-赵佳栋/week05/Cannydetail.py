#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：Cannydetail.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/18 23:26
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


if  __name__ == '__main__':
    # 1、灰度化
    img = cv2.imread("../lenna.png", 1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # pic_path = '../lenna.png'
    # img = plt.imread(pic_path)
    # print("image", img)
    # if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
    #     img = img * 255  # 还是浮点数类型
    # img_gray = img.mean(axis=-1)

    plt.figure(1)
    plt.imshow(img_gray.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 2、高斯平滑
    # 2.1 生成高斯核
    sigma = 0.5 #高斯平滑时的高斯核参数，标准差，可调
    dim = 5 #高斯核单位尺寸是5
    gaosi_filter = np.zeros([dim,dim])
    tmp = [i - dim// 2 for i in range(dim)]
    n1 = 1/(2 * math.pi * sigma**2)
    n2 = -1/(2 * sigma**2)
    for i in range(dim):
        for j in range(dim):
            gaosi_filter[i,j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    gaosi_filter = gaosi_filter / gaosi_filter.sum()
    # 2.2 使用高斯核 进行滤波
    dx,dy = img_gray.shape
    img_new = np.zeros(img_gray.shape)
    tmp = dim//2
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant')

    for i in range(dx):
        for j in range(dy):
            img_new[i,j] = np.sum (img_pad[i:i+dim,j:j+dim] * gaosi_filter)

    plt.figure(2)
    plt.imshow(img_new.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 3、检测边缘（通过sobel梯度算子求梯度）
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    tidu_x = np.zeros(img_new.shape)
    tidu_y = np.zeros(img_new.shape)
    tidu = np.zeros(img_new.shape)
    img_pad = np.pad(img_new,((1,1),(1,1)),'constant')
    for i in range(dx):
        for j in range(dy):
            tidu_x[i,j]= np.sum(img_pad[i:i+3,j:j+3] * sobel_x )
            tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3] * sobel_y )
            tidu[i,j] = np.sqrt(tidu_x[i,j]**2 + tidu_y[i,j]**2)

    tidu_x[tidu_x == 0 ] = 0.00000001
    tan_sita = tidu_y / tidu_x

    plt.figure(3)
    plt.imshow(tidu.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 4、非极大值抑制
    # 通过 tan_sita 来判断像素点的梯度方向范围，然后找出像素点同梯度方向的两个像素值 num1 num2 的梯度大小，最后筛选出梯度最大的像素点
    img_yizhi = np.zeros(tidu.shape)
    for i in range(1,img_yizhi.shape[0] - 1):
        for j in range(1,img_yizhi.shape[1] - 1):
            flag = True
            temp = tidu[i-1:i+2,j-1:j+2]  # 当前像素点的八邻域矩阵的梯度
            if tan_sita[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / tan_sita[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / tan_sita[i, j] + temp[2, 1]
                if not (tidu[i, j] > num_1 and tidu[i, j] > num_2):
                    flag = False
            elif tan_sita[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / tan_sita[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / tan_sita[i, j] + temp[2, 1]
                if not (tidu[i, j] > num_1 and tidu[i, j] > num_2):
                    flag = False
            elif tan_sita[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * tan_sita[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * tan_sita[i, j] + temp[1, 0]
                if not (tidu[i, j] > num_1 and tidu[i, j] > num_2):
                    flag = False
            elif tan_sita[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * tan_sita[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * tan_sita[i, j] + temp[1, 2]
                if not (tidu[i, j] > num_1 and tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = tidu[i, j]

    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    # 5、 双阈值检测
    #  5.1遍历im_yizhi的像素点，高于高阈值的是强边缘直接保留（赋值255），低于低阈值的直接舍弃（赋值0），
    low_bound = tidu.mean()* 0.5 # 阈值设置
    high_bound = low_bound * 3  # 阈值设置
    zhan = [] #设置一个空数组来模拟栈
    for i in range(1,img_yizhi.shape[0]-1):
        for j in range(1,img_yizhi.shape[1]-1):
            if img_yizhi[i,j] <= low_bound:
                img_yizhi[i,j] = 0
            elif img_yizhi[i,j] >=high_bound:
                img_yizhi[i,j] = 255
                zhan.append([i,j]) # 把边缘坐标入栈


    # 5.2 通过上面的流程，还剩介于双阈值之间的像素需要操作
    # 从强边缘出发去寻找标记强边缘附近的边缘，若介于阈值之间，则标记为强边缘，并把该像素坐标入栈
    while not len(zhan) == 0:
        # 先取出强边缘坐标，以这个坐标为中心找出其8邻域像素，然后判断8邻域像素有没有介于双阈值之间的
        # 介于双阈值之间的直接标记为强边缘
        temp1,temp2 = zhan.pop()

        a = img_yizhi[temp1-1:temp1+2 , temp2-1:temp2+2] # 取8邻域切片
        if (a[0,0] > low_bound) and (a[0,0] < high_bound):
            img_yizhi[temp1 - 1,temp2 - 1] = 255
            zhan.append([temp1 - 1, temp2 - 1])

        if (a[0,1] > low_bound) and (a[0,1] < high_bound):
            img_yizhi[temp1  , temp2 ] = 255
            zhan.append([temp1 - 1, temp2])

        if (a[0,2] > low_bound) and (a[0,2] < high_bound):
            img_yizhi[temp1 - 1, temp2 + 1] = 255
            zhan.append([temp1 - 1, temp2 + 1])

        if (a[1,0] > low_bound) and (a[1,0] < high_bound):
            img_yizhi[temp1 , temp2 - 1] = 255
            zhan.append([temp1 , temp2 - 1])

        if (a[1,2] > low_bound) and (a[1,2] < high_bound):
            img_yizhi[temp1 , temp2 + 1] = 255
            zhan.append([temp1 , temp2 + 1])

        if (a[2,0] > low_bound) and (a[2,0] < high_bound):
            img_yizhi[temp1 + 1, temp2 - 1] = 255
            zhan.append([temp1 + 1, temp2 - 1])

        if (a[2,1] > low_bound) and (a[2,1] < high_bound):
            img_yizhi[temp1 + 1, temp2] = 255
            zhan.append([temp1 + 1, temp2])

        if (a[2,2] > low_bound) and (a[2,2] < high_bound):
            img_yizhi[temp1 + 1, temp2 + 1] = 255
            zhan.append([temp1 + 1, temp2 + 1])

    # 5.3 剩下的点是从边缘出发循环检测也连接不到的点，直接过滤掉
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i,j] != 0 and img_yizhi[i,j] != 255:
                img_yizhi[i,j] = 0

    plt.figure(5)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()