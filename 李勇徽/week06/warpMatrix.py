import cv2
import numpy as np


def getWarpMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    sz = src.shape[0]
    A = np.zeros((sz * 2, 8))
    B = np.zeros((sz * 2, 1))
    # 填充A和B矩阵
    for i in range(0, sz):
        A_i = src[i, :] # 原点
        B_i = dst[i, :] # 目标点
        A[i * 2, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        A[i * 2 + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[i * 2] = B_i[0]
        B[i * 2 + 1] = B_i[1]
    A = np.mat(A)
    # A * warpMatrix = B
    warpMatrix = A.I * B
    # 先转置
    warpMatrix = np.array(warpMatrix).T[0]
    # 插入a33 = 1
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], 1, 0)
    warpMatrix = warpMatrix.reshape((3,3))
    print(warpMatrix)
    return warpMatrix

if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpMatrix = getWarpMatrix(src, dst)
    img_wp = cv2.warpPerspective(img, warpMatrix, (337, 488))
    # cv2.imshow("原图", img)
    cv2.imshow("透视变换后", img_wp)
    cv2.waitKey(0)

