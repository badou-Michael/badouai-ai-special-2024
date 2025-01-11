import numpy as np

"""
a11x+a12y+a13-a31xX'-a32X'y = X'
a21x+a22y+a23-a31xY'-a32yY' = Y' 8个参数
源点四个坐标分别为A：（x0,y0),(x1,y1),(x2,y2),(x3,y3) 
目标点四个坐标分别为B：(X'0,Y'0),(X'1,Y'1),(X'2,Y'2),(X'3,Y'3)
"""


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]  # 位置点个数
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    for i in range(nums):
        A_i = src[i, :]  # 第i行所有的列
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
        print(f"A_i={A_i}, B_i={B_i}, A[2 * i, :]={A[2 * i, :]},  B[2 * i]={B[2 * i]}, "
              f"A[2 * i + 1, :]={A[2 * i + 1, :]},  B[2 * i + 1]={B[2 * i + 1]}")

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    print(f"warpMatrix_before={warpMatrix}")
    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    print(f"warpMatrix_after={warpMatrix}")
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
