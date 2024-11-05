import numpy as np
import cv2

def WarpPerspectiveMatrix(src, dst):
    '''
    利用原图需要变换的物体的四个顶点坐标和变换后的四个顶点坐标求出变换矩阵warpMatrix
    A * warpMatrix = B
    :param src:原图需要变换物体的四个顶点
    :param dst:新图对应的四个顶点
    :return:变换矩阵
    '''
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        # 2*8矩阵
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出变换矩阵warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32 共8个参数

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

def tsbh(warpMatrix, dst, img):
    w = int(dst[1,0]-dst[0,0])+1
    h = int(dst[2,1]-dst[0,1])+1
    result = np.zeros((h,w,3), np.uint8)
    W = np.mat(warpMatrix)
    W = W.I
    for i in range(w):
        for j in range(w):
            XY1 = np.array([[j],[i],[1]])
            XY1 = np.mat(XY1)
            x,y, _= W.dot(XY1)
            # 验证算出来的坐标是否和src里的相对应
            if i == h-1 and j ==0:
                print(x, y)
            # 算出的索引超出原图大小时，令其等于边界
            if y >=960:
                y = 959
                pass
            if y < 0:
                y = 0
                pass
            if x > 540:
                x = 539
                pass
            if x < 0:
                x = 0
                pass
            result[i, j, 0] = img[int(y), int(x), 0]
            result[i, j, 1] = img[int(y), int(x), 1]
            result[i, j, 2] = img[int(y), int(x), 2]
    return result


if __name__ == '__main__':
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    img_src = cv2.imread('photo1.jpg', 1)
    print(img_src.shape)
    img = np.copy(img_src)
    res = tsbh(warpMatrix, dst, img)
    cv2.imshow('img_src', img_src)
    cv2.imshow('res', res)
    cv2.waitKey(0)






##################
'''
# 直接cv2接口获取
import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()

'''
# 注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = WarpPerspectiveMatrix(src, dst)
print("warpMatrix:")
print(m)
result = WarpPerspectiveMatrix(result3, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)

'''