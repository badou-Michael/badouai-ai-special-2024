import numpy as np

def WarpMatrix(src, dst):
    nums = src.shape[0]
    Matrix_a = np.zeros((nums*2, 8))
    Matrix_b = np.zeros((nums * 2, 1))
    for i in range(0, nums):
        Matrix_a[i*2, :] = [src[i, 0], src[i, 1], 1, 0, 0, 0, -src[i, 0] * dst[i, 0], -src[i, 1] * dst[i, 0]]
        Matrix_a[i*2+1, :] = [0, 0, 0, src[i, 0], src[i, 1], 1, -src[i, 0] * dst[i, 1], -src[i, 1] * dst[i, 1]]
        Matrix_b[i*2, :] = [dst[i, 0]]
        Matrix_b[i*2+1, :] = [dst[i, 1]]
    A = np.mat(Matrix_a)
    warpMatrix = A.I * Matrix_b
    wM = np.insert(warpMatrix, 8, 1, axis=0)
    wM_shape = wM.reshape((3, 3))
    return wM_shape

if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    ws = WarpMatrix(src, dst)
    print(ws)
