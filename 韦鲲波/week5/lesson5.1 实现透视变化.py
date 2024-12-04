import numpy as np
import random as rm
from random import randint

source = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
source = np.array(source)

target = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
target = np.array(target)

war = np.zeros((3, 3))

def warpMatrix(source, target):
    # 纠错
    assert source.shape[0], '行列不一样'
    assert source.shape[0] >= 4, '源图不够大'

    # 设置源图矩阵和目标图矩阵
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))

    # 计算warpMatrix
    for i in range(4):
        A[2*i,:] = [
            source[i][0],
            source[i][1],
            1,
            0,
            0,
            0,
            -source[i][0]*target[i][0],
            -source[i][1]*target[i][0],
        ]

        A[2*i+1,:] = [
            0,
            0,
            0,
            source[i][0],
            source[i][1],
            1,
            -source[i][0] * target[i][1],
            -source[i][1] * target[i][1],
        ]

        B[2*i,:] = target[i][0]
        B[2*i+1,:] = target[i][1]

    result = np.linalg.solve(A, B)
    result = np.insert(result, result.shape[0], 1)
    return result.reshape(3, 3)

a = warpMatrix(source, target)
print(a)













