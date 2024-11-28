import cv2
import numpy as np

def WarpPerspectiveMatrix(src, dst):
    # 断言条件为真，正常执行
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4  # 确保源和目标点数量相同且至少为4
    nums = src.shape[0]  # 获取点的数量
    print(nums)
    # 创建一个2*nums x 8的二维numpy数组，用于存储方程的系数
    A = np.zeros((2*nums, 8))  # A*warpMatrix=B
    B = np.zeros((2*nums, 1))  # 创建一个2*nums x 1的数组，用于存储目标点的坐标
    for i in range(0, nums):
        A_i = src[i, :]  # 获取源点的坐标
        B_i = dst[i, :]  # 获取目标点的坐标
        # 填充A的偶数行
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                     -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]  # 填充B的偶数行
        
        # 填充A的奇数行
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]  # 填充B的奇数行
 
    A = np.mat(A)  # 将A转换为矩阵形式
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 计算变换矩阵
    
    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]  # 转换为一维数组
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))  # 重塑为3x3矩阵
    return warpMatrix  # 返回变换矩阵

# 读取图像文件
img = cv2.imread('photo1.jpg')

# 创建图像的副本以进行透视变换
result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
# 定义源点和目标点的坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

# 打印原始图像的形状
print(img.shape)

# 生成透视变换矩阵；进行透视变换
m = WarpPerspectiveMatrix(src, dst)
#m = cv2.getPerspectiveTransform(src, dst)  # 计算从源点到目标点的透视变换矩阵
print("warpMatrix:")  # 打印透视变换矩阵
print(m)

# 应用透视变换
result = cv2.warpPerspective(result3, m, (337, 488))  # 使用变换矩阵对图像进行透视变换

# 显示原始图像和变换后的图像
cv2.imshow("src", img)  # 显示原始图像
cv2.imshow("result", result)  # 显示透视变换后的图像

# 等待按键事件
cv2.waitKey(0)  # 等待用户按下任意键
