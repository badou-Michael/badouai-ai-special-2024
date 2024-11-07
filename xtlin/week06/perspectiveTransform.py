import cv2
import numpy as np

# 1
# img = cv2.imread("photo1.jpg")
# ret2 = img.copy()
#
# src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# dst = np.float32([[207,151], [337+207, 0+151], [0+207, 488+151], [337+207, 488+151]])
# print(img.shape)
#
# warpMatrix = cv2.getPerspectiveTransform(src, dst)
# print("warpMatirx:")
# print(warpMatrix)
#
# ret = cv2.warpPerspective(ret2, warpMatrix, (540, 960))
# cv2.imshow("src", img)
# cv2.imshow("ret", ret)
# cv2.waitKey(0)

# 2
def warpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    n = src.shape[0]
    A = np.zeros((2*n, 8))  # <class 'numpy.ndarray'>
    B = np.zeros((2*n, 1))
    for i in range(n):
        s_i = src[i]
        d_i = dst[i]
        A[2*i] = [s_i[0], s_i[1], 1, 0, 0, 0, -s_i[0]*d_i[0], -s_i[1]*d_i[0]]
        A[2*i+1] = [0, 0, 0, s_i[0], s_i[1], 1, -s_i[0]*d_i[1], -s_i[1]*d_i[1]]
        B[2*i] = d_i[0]
        B[2*i+1] = d_i[1]

    # print(type(A))  # A: <class 'numpy.ndarray'>
    A = np.mat(A)  # A after np.nat: <class 'numpy.matrix'>
    warpMatrix = A.I * B  # A^(-1) * B

    # 矩阵后处理
    # print(type(warpMatrix))  # warpMatrix: <class 'numpy.matrix'>

    # warpMatrix:
    # (8, 1)
    # [[-5.01338334e-01]
    #  [-1.35357643e+00]
    #  [5.82386716e+02]
    #  [-1.66533454e-15]
    #  [-4.84035391e+00]
    #  [1.38781980e+03]
    #  [-4.33680869e-19]
    #  [-4.14856327e-03]]

    # np.array(warpMatrix).T:
    # (1, 8)
    # [[-5.01338334e-01 - 1.35357643e+00  5.82386716e+02 - 1.66533454e-15
    #   - 4.84035391e+00  1.38781980e+03 - 4.33680869e-19 - 4.14856327e-03]]

    # np.array(warpMatrix).T[0]:
    # (8,)
    # [-5.01338334e-01 - 1.35357643e+00  5.82386716e+02 - 1.66533454e-15
    #  - 4.84035391e+00  1.38781980e+03 - 4.33680869e-19 - 4.14856327e-03]

    # *** if np.array(warpMatrix).T -> np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    # (2, 8)
    # [[-5.01338334e-01 - 1.35357643e+00  5.82386716e+02 - 1.66533454e-15
    #   - 4.84035391e+00  1.38781980e+03 - 4.33680869e-19 - 4.14856327e-03]
    #  [1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00
    #  1.00000000e+00  1.00000000e+00  1.00000000e+00  1.00000000e+00]]

    warpMatrix = np.array(warpMatrix).T[0]  # warpMatrix after np.array: <class 'numpy.ndarray'>
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # shape: (9,)
    warpMatrix = warpMatrix.reshape((3, 3))

    return warpMatrix

if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = warpPerspectiveMatrix(src, dst)
    print("warpMatrix: ")
    print(warpMatrix)
