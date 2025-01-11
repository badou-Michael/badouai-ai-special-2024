import numpy as np


def projecttiveMatrix(src,dst):
    """
    投影矩阵
    @param src: 原四个点
    @param dst: 目标4个点
    @return:
    """
    #初始化一个8x8矩阵
    left_matrix = np.zeros((8,8))
    #等号右边的矩阵，也就是8个点的坐标
    right_matrix = np.zeros((8,1))
    #要求projecttiveMatrix，8x8矩阵分为四组
    for i in range(4):
        src_n = src[i,:] #源点8x8矩阵的第i行
        dst_n = dst[i,:] #目标点8x8矩阵的第i行
        #4个点对应的坐标
        xi,yi = src_n[0],src_n[1]
        xi_1,yi_1 = dst_n[0],dst_n[1]
        right_matrix[2*i,:] = [xi_1]
        right_matrix[2*i+1,:] = [yi_1]
        left_matrix[2*i,:] = [xi,yi,1,0,0,0,-xi*xi_1,-yi*xi_1]
        left_matrix[2*i+1,:] = [0,0,0,xi,yi,1,-xi*yi_1,-yi*yi_1]
    #转为矩阵，方便求逆矩阵进行矩阵的除法
    left_matrix = np.mat(left_matrix)
    #根据A*warpMatrix=B
    warpMatrix = left_matrix.I * right_matrix
    # warpMatrix1 = np.copy(warpMatrix)
    #矩阵转为列表，然后转置后添加a33为1
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    # warpMatrix1 = np.vstack((warpMatrix1,1))
    # warpMatrix1 = warpMatrix1.reshape((3,3))
    # print(warpMatrix1)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    warpMatrix = projecttiveMatrix(src, dst)