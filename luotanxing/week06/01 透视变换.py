import cv2
import numpy as np






def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    # A为8*8矩阵
    A = np.zeros((2 * nums, 8))
    # B为8*1矩阵
    B = np.zeros((2 * nums, 1))
    '''
      a11*x + a12*y + a13 = a31*x*X + a32*X*y + X    (a33=1)
      a21*x + a22*y + a23 = a31*x*Y + a32*Y*y + Y    (a33=1)
      
      a11*x + a12*y + a13 - a31*x*X - a32*X*y = X
      a21*x + a22*y + a23 - a31*x*Y - a32*Y*y = Y
      
      A*warpMatrix=B
      
      A = [[x,y,1,0,0,0,-x*X,-X*y],    wrap = [a11,a12,a13,a21,a22,a23,a31,a32,a33]          B = [X,Y]
           [0,0,0,x,y,1,-x*Y,-Y*y],]
    '''
    for i in range(0, nums):
        A_i = src[i, :]   # (x,y)
        B_i = dst[i, :]   # (X,Y)
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,-A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    A = np.mat(A)
    print(A.shape)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    # warpMatrix=A^-1*B
    warpMatrix =  np.linalg.inv(A) * B
    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')

    result3 = img.copy()
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    # 生成透视变换矩阵；进行透视变换
    m = WarpPerspectiveMatrix(src, dst)
    print("warpMatrix:")
    print(m)
    # 337 448 为尺寸  A*warpMatrix=B
    result = cv2.warpPerspective(result3, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)
