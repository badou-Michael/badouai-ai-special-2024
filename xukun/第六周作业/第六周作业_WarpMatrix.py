import cv2
import numpy as np


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))

    for i in range(nums):
        A_I = src[i, :]
        B_I = dst[i, :]
        A[2 * i, :] = [A_I[0], 0, A_I[1], 0, 1, 0, -A_I[0] * B_I[0], -A_I[1] * B_I[0]]
        A[2 * i + 1, :] = [0, A_I[0], 0, A_I[1], 0, 1, -A_I[0] * B_I[1], -A_I[1] * B_I[1]]
        B[2 * i] = B_I[0]
        B[2 * i + 1] = B_I[1]
    A = np.mat(A)

    warpMatrix = A.T*B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], 1)
    warpMatrix = np.reshape(warpMatrix, (3, 3))
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
