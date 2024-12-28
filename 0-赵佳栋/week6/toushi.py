#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：toushi.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/11/1 18:33
'''
import cv2
import numpy as np
#Warp Perspective Matrix ： 弯曲透视矩阵

def warpPerspectiveMatrix(src, dst):
    """
    计算透视变换矩阵。

    该函数基于给定的源坐标点集（src）和目标坐标点集（dst），通过构建线性方程组并求解，得到透视变换矩阵。

    参数:
    src (numpy.ndarray): 源坐标点集，形状应为 (n, 2)，其中n >= 4，表示至少有4个点，每个点包含x和y坐标。
    dst (numpy.ndarray): 目标坐标点集，形状与src一致，对应src中各点变换后的目标坐标。

    返回:
    warpMatrix: 计算得到的3x3透视变换矩阵。

    注意：
    函数内部会进行断言检查，确保src和dst的点数相同且点数不少于4个。
    """
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    points = src.shape[0]

    # 构建线性方程组 A * warpMatrix = B  设置AB两个空矩阵  A是系数矩阵   B是结果矩阵
    # 初始化AB矩阵大小
    A = np.zeros((2 * points, 8))
    B = np.zeros((2 * points, 1))

    for i in range(0 , points):
        A_i = src[i, :]  #取第 i 行数据，即 x坐标 y坐标  代表Xi  Yi
        B_i = dst[i, :]  # 代表 X'i   Y'i

        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,  -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        A[2*i+1,:] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1] ]
        B[2*i] = B_i[0]
        B[2*i+1] = B_i[1]
    # 先 A 转换为 numpy 的矩阵类型（np.mat），然后矩阵求逆（A.I）和 B 相乘，透视变换矩阵 warpMatrix
    A = np.mat(A)
    warpMatrix = A.I * B

    # 此时得到的warpMatrix是个列向量，有a11~a32 8列 ，还需要添加a33
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    # reshape() 函数是 numpy 中改变数组形状的方法。这里的参数 (3, 3) 表示将当前的 warpMatrix 数组重塑为一个 3×3 的二维矩阵形式
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    print('warpMatrix')
    src = [[1100.0, 760.0], [3002.0, 770.0], [419.0, 2404.0], [3935.0, 2324.0]]
    src = np.array(src)

    dst = [[419.0, 760.0], [3935.0, 760.0], [419, 2404.0], [3935.0, 2404.0]]
    dst = np.array(dst)

    warpMatrix = warpPerspectiveMatrix(src, dst)
    print(warpMatrix)

    img = cv2.imread('./IMG_2861.JPG')
    height, width = img.shape[:2]
    # 使用透视变换矩阵对图片进行变换
    warped_img = cv2.warpPerspective(img, warpMatrix, (height, width))

    # 显示原始图片和变换后的图片（可以按需保存图片等操作）
    cv2.imshow("Original Image", img)
    cv2.imshow("Warped Image", warped_img)
    cv2.waitKey(0)











