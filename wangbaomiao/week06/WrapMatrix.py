# -*- coding: utf-8 -*-
# time: 2024/10/28 11:17
# file: WrapMatrix.py
# author: flame
import numpy as np


def WrapPerspectionMatrix(src, dst):
    """
    计算透视变换矩阵。

    该函数根据源点集和目标点集，计算出透视变换矩阵。
    源点集和目标点集都应包含4个点，且点的顺序对应。

    参数:
    src: numpy数组，包含4个源点坐标。
    dst: numpy数组，包含4个目标点坐标。

    返回:
    wrapMatrix: 3x3的透视变换矩阵。
    """
    # 确保源点集和目标点集的长度均为4
    assert src.shape[0] == dst.shape[0] and src.shape[0] == 4, "源点集和目标点集必须包含4个点"

    # 获取点的数量
    num = src.shape[0]

    # 初始化A矩阵和B矩阵
    # A矩阵用于构建线性方程组，B矩阵用于存储目标点的坐标
    A = np.zeros((2 * num, 8))
    B = np.zeros((2 * num, 1))

    # 遍历每个点，构建A矩阵和B矩阵
    for i in range(0, num):
        A_i = src[i, :]  # 获取当前源点坐标
        B_i = dst[i, :]  # 获取当前目标点坐标

        # 构建A矩阵的第2i行
        # 这一行用于表示x方向的变换关系
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]

        # 构建A矩阵的第2i+1行
        # 这一行用于表示y方向的变换关系
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]

        # 构建B矩阵的第2i行和第2i+1行
        B[2*i] = B_i[0]  # 目标点的x坐标
        B[2*i+1] = B_i[1]  # 目标点的y坐标

    # 将A转换为矩阵类型，以便进行矩阵运算
    A = np.mat(A)

    # 计算透视变换矩阵
    # 使用A的逆矩阵乘以B，得到透视变换矩阵的前8个元素
    wrapMatrix = A.I * B

    # 将计算结果转换为数组形式
    wrapMatrix = np.array(wrapMatrix).T[0]

    # 添加第9个元素1，构成完整的3x3透视变换矩阵
    wrapMatrix = np.insert(wrapMatrix, wrapMatrix.shape[0], values=1.0, axis=0)

    # 将透视变换矩阵 reshape 为 3x3 的形式
    wrapMatrix = wrapMatrix.reshape((3, 3))

    # 返回计算得到的透视变换矩阵
    return wrapMatrix


if __name__ == '__main__':
    print("WrapMatrix")
    # 生成四个像素点矩阵
    src = np.array([[10.0, 450.0], [130.0, 420.0], [550.0, 470.0], [380.0, 500.0]])
    dst = np.array([[10.0, 992.0], [130.0, 10.0], [130.0, 130.0], [10.0, 920.0]])

    # 调用函数计算透视变换矩阵
    wrapMatrix2 = WrapPerspectionMatrix(src, dst)

    # 打印计算得到的透视变换矩阵
    print(wrapMatrix2)
