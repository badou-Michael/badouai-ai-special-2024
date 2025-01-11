"""
    6.1warpMatrix
"""

import numpy as np

def WarpPerspectiveMatrix(src, dst):
    # 确保源点和目标点的数量至少为 4 个且数量相等
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    # 创建一个 2*nums 行 8 列的全零矩阵 A，用于构建方程组的系数矩阵
    A = np.zeros((2*nums, 8))
    # 创建一个 2*nums 行 1 列的全零矩阵 B，用于构建方程组的常数项矩阵
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        # 构建方程组的第一部分，对应 x 坐标的方程
        #矩阵A的第2*i行所有列的元素
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]

        # 构建方程组的第二部分，对应 y 坐标的方程
        #矩阵A的第2*i + 1行所有列的元素
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]

    # 将 A 转换为矩阵类型
    A = np.mat(A)
    # 通过 A 的逆矩阵与 B 相乘来求解方程组，得到变换矩阵的前 8 个元素
    warpMatrix = A.I * B
    # 后处理部分
    # 将一维数组转换为一维矩阵并转置，获取变换矩阵的前 8 个元素
    warpMatrix = np.array(warpMatrix).T[0]
    # 在变换矩阵的末尾插入元素 1，得到完整的 3x3 变换矩阵
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    # 将一维数组重新调整为 3x3 的矩阵形式
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    # 定义源点坐标列表并转换为 NumPy 数组
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    # 将输入的列表src转换为NumPy数组。
    src = np.array(src)

    # 定义目标点坐标列表并转换为 NumPy 数组
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    # 将输入的列表src转换为NumPy数组。
    dst = np.array(dst)

    # 调用函数计算透视变换矩阵
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
