import cv2
import matplotlib.pyplot as plt
import numpy as np


def WarpPerspectiveMatrix(src, dst):
    if src.shape[0] != dst.shape[0] and src.shape[0] != 4:
        raise ValueError('input error!')

    nums = src.shape[0]
    A = np.zeros((nums * 2, 8))
    B = np.zeros((nums * 2, 1))

    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[i * 2, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[i * 2, :] = [B_i[0]]
        A[i * 2 + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[i * 2 + 1, :] = [B_i[1]]

    A = np.mat(A)
    m = A.I * B
    m = np.insert(m, m.shape[0], 1)
    return m.reshape((3, 3))


if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    m = WarpPerspectiveMatrix(src, dst)
    result = cv2.warpPerspective(img, m, (337, 488))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.show()
