import numpy as np
import cv2
#求透视变换矩阵，参数有源点，目标点
def WarpPerspectiveMatrix(src,dst):
    #先检查输入的参数是否正确，目标点和源点数量要一致，还有必须大于或等于四个
    assert src.shape[0] == dst.shape[0] and src.shape[0]>=4
    #接下来根据方程组求解透视变换矩阵，Ax=B, 在这里未知数是透视变换矩阵的八个参数，先求矩阵A，B，然后解方程，求出x
    #先把建立空的A，B矩阵
    num = src.shape[0]
    A = np.zeros((num*2, num*2))
    B = np.zeros(num*2)
    for i in range(num):
        src_x, src_y = src[i, :]
        dst_x, dst_y = dst[i, :]
        A[i*2, :] = [src_x, src_y, 1, 0, 0, 0, -src_x*dst_x, -src_y*dst_x]
        B[i*2]=dst_x
        A[i*2+1, :] = [0, 0, 0, src_x, src_y, 1, -src_x*dst_y, -src_y*dst_y]
        B[i*2+1] =dst_y
    warpMatrix = np.linalg.solve(A, B)
    warpMatrix = warpMatrix.reshape(-1)  # 将结果从列向量转换为行向量
    warpMatrix = np.insert(warpMatrix, 8, 1.0)  # 插入 a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))  # 重塑为3x3矩阵

    return warpMatrix


if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
