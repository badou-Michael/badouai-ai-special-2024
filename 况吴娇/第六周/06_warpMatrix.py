##没使用 OpenCV 的 getPerspectiveTransform 函数
'''
脚本中，透视变换矩阵是通过解决以下线性方程组来计算的
A × warpMatrix = B  ( 透视变换矩阵 warpMatrix)
在透视变换中，warpMatrix 是一个 3x3 的矩阵，它包含了变换所需的所有参数。
其中，A33通常被设置为 1，因为这是一个齐次坐标变换。这个矩阵的每一列代表一个变换参数，它们共同定义了如何将图像中的点从一个位置映射到另一个位置
'''

import numpy as np

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4 #这行代码使用 assert 语句确保 src 和 dst 中的点的数量相同，并且至少有4个点。这是因为透视变换至少需要4对对应点来唯一确定变换矩阵。
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B 那么 warpMatrix = A.I * B ;I⋅warpMatrix=A^(−1)⋅ B
    B = np.zeros((2*nums, 1))
    ##A = np.zeros((2*nums, 8)) 创建了一个零矩阵，这是因为 np.zeros 是 NumPy 库中的一个函数，它用于生成一个由零组成的数组。
    # 这个函数接受一个元组作为参数，指定了数组的形状。这里的 2*nums 和 8 分别是矩阵的行数和列数
    #2 * nums：表示矩阵的行数。因为每个点（由nums表示）将产生两个方程（一个用于x坐标，一个用于y坐标），
    # 所以行数是点的数量的两倍。8：表示矩阵的列数。这是因为每个方程需要8个系数来表示透视变换矩阵中的未知数（a11, a12, a13, a21, a22, a23, a31, a32）。
    #在透视变换中，每个点对（源点和目标点）会产生两个方程，因为目标点的坐标（x, y）是由源点的坐标通过变换矩阵计算得到的。这些方程可以表示为：
    '''
    a11x1+a12y1+a13=x1
    a21x1+a22y1+a23=y1
    ....=>
    a11xn+a12yn+a13=Xn
    a21xn+a22yn+a23=Yn
    '''
    for i in range(0, nums):
        A_i = src[i,  :]
        B_i = dst[i,  :]
        A[ 2* i, :] = [A_i[0],A_i[1], 1, 0, 0, 0,- A_i[0]*B_i[0],-A_i[1]*B_i[0]]
        ##A_i[0]：A_i[0]：xi的系数;A_i[1]：y i的系数;1：常数项 a13的系数-A_i[0]*B_i[0],-A_i[1]*B_i[0] a31,32的系数（与 Xi项相关）
        B[ 2* i]=B_i[0]
        A[2 * i+1, :]= [0, 0, 0,A_i[0], A_i[1], 1,-A_i[0] * B_i[1], -A_i[1] * B_i[1]]

        B[2 *i  + 1] = B_i[1] ##这个循环构建了一个线性方程组 A×warpMatrix=B



    A=np.mat(A) #np.mat  ## A 转换为 NumPy 的矩阵类型
    # #np.mat 是用来创建矩阵对象的函数。这个函数属于 NumPy 的一个子模块 numpy.matrix，它提供了一个矩阵类，用于存储二维数组。
    warpMatrix = A.I * B
    ##以使用 np.dot() 或 @ 运算符来进行矩阵乘法。
    ###求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    warpMatrix = np.array(warpMatrix).T[0]
    # 转置warpMatrix。转置矩阵意味着交换它的行和列。
    # 通过索引[0]提取转置矩阵的第一行。
    ##转置是因为求出来的是列矩阵，要转置成3*3
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    ##numpy.insert(arr, obj, values, axis=None) #如果是整数数组，表示多个插入位置的索引。在特定的行位置插入一行，你会设置 axis=0。
    #对于一维数组，np.insert 的 axis 参数应该是 None 或省略不写，这样 NumPy 会将数组视为一维并沿着这个维度进行插入
    warpMatrix = warpMatrix.reshape((3, 3))  # 重塑为 3x3 矩阵
    return warpMatrix


if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)

