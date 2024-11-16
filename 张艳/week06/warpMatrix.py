import numpy as np
 
def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4 # 一个断言语句，用于在代码中进行条件检查。如果条件不满足，程序将抛出一个 AssertionError 并终止执行。
    
    nums = src.shape[0] # 第一个维度（行数）
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B，(2*nums, 8)是传递给函数的参数的圆括号，它是一个元组，用于定义要创建的数组的形状。
    B = np.zeros((2*nums, 1))
    for i in range(0, nums): # 左闭右开
        A_i = src[i,:] # src 中第 i 行的所有元素
        B_i = dst[i,:]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]        
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
 
    A = np.mat(A) # 将一个普通的二维数组 A 转换为一个矩阵对象
    #用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    
    #之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0] # 选择转置后数组的第一行
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1，axis=0 表示沿着第一个轴（行）插入。最后一行之后插入。
    warpMatrix = warpMatrix.reshape((3, 3)) #一维数组重新排列成一个 3x3 的二维矩阵
    return warpMatrix
 
if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)    
    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)
    
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
    
'''
np.array（）和np.float32（）都生成数组，只是数组里的数据类型不同，前者的数据类型灵活且精度高
'''
