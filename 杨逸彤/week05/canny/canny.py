import math

import cv2

# 对图像进行灰度化
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("d:\\Users\ls008\Desktop\lenna.png",0)

# 对图像进行高斯滤波
# 准备滤波器
sigma = 0.5
dim = 5
Gaussian_filter = np.zeros((dim,dim))
tmp = [i - dim // 2 for i in range(dim)] #生成以dim为中间，两边对称的序列
# 计算高斯核，其中math.pi是π，sigma ** 2是方差
n1 = 1 / (2 * math.pi * sigma ** 2) # 表示1/(2πσ²)，常数因子，用于定义高斯滤波器的强度
n2 = -1 / (2 * sigma**2) # 表示-1/(2σ²)，负常数，与变量的平方相乘以形成一个指数项
for i in range(dim):
        for j in range(dim):
                # 确定高斯滤波（i,j)位置的权重值
                Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2)) # math.exp表示对输入进行指数运算，生成一个权重值
# 对整个滤波器进行归一化，以确保其应用后不会改变输入图像的亮度
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
img_new = np.zeros(img.shape)
# 获取图片大小
dx,dy = img.shape
# 获取滤波中心点
tmp = dim//2
# 边缘填补
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant') # 表示用constant常数的形式对img图片行列各向外填充tmp个像素
#卷积操作
for i in range(dx):
        for j in range(dy):
                # mg_pad[i:i+dim, j:j+dim]：提取填充图像中的一个局部区域，i:i+dim 是切片操作,表示从 i 行开始，取到 i + dim 行（不包括 i + dim）,列同理
                # * Gaussian_filter：将每个区域内的像素值与对应的滤波器权重相乘
                # np.sum(...)：对逐元素乘积的结果进行求和，得到输出图像在(i, j)位置的像素值。这个值是该区域所有加权后的像素值的总和
                img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
plt.figure(1)
plt.imshow(img_new.astype(np.uint8), cmap='gray')
plt.axis('off')

# 检测图像中的水平、垂直和对角边缘
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape) # 存储x方向梯度
img_tidu_y = np.zeros([dx, dy]) # 存储y方向梯度
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
# 使用Sobel核计算x和y方向的梯度，并将幅值存储到img_tidu中
for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
# img_tidu_x中有像素值为0的时候将0变成极小值，避免后续计算出现除零错误
img_tidu_x[img_tidu_x == 0] = 0.00000001
# 获取梯度方向，用于后续非极大值的计算
angle = img_tidu_y/img_tidu_x
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray') #img_tidu.astype(np.uint8) 转换为 uint8
plt.axis('off')

# 对梯度幅值进行非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx-1): # -1是为了处理边界问题
    for j in range(1, dy-1):
        flag = True  # 用于标记当前像素点是否为局部最大值
        temp = img_tidu[i-1:i+2, j-1:j+2]  # 取出当前i,j的临近值，形成3*3矩阵
        # angle[i, j]是(i, j)像素的梯度
        if angle[i, j] <= -1:# 如果梯度方向小于等于-1
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1] # 通过双线性插法计算dTmp1
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1] # 通过双线性插法计算dTmp2
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2): # 通过dTmp1、dTmp2和梯度值算出是否是极大值
                flag = False
        elif angle[i, j] >= 1:# 如果梯度方向大于等于1
            num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:# 如果梯度方向大于0
            num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:# 如果梯度方向小于0
            num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            #如果是极大值，赋值到img_yizhi对应为止
            img_yizhi[i, j] = img_tidu[i, j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')

# 双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
lower_boundary = img_tidu.mean() * 0.5 # 计算低阈值
high_boundary = lower_boundary * 3 # 计算高阈值，设计为高阈值的3倍
zhan = [] # 新建一个空列表用于存储满足条件的坐标
# 双阈值检测
for i in range(1, img_yizhi.shape[0] - 1):  # -1是为了处理边缘像素的问题
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high_boundary:  # 获取大于等于高阈值的像素，变为255
            img_yizhi[i, j] = 255
            zhan.append([i, j])
        elif img_yizhi[i, j] <= lower_boundary:  # 获取小于低阈值的像素，变为0
            img_yizhi[i, j] = 0

# 边缘检测
while not len(zhan) == 0: # 如果列表不为0，循环列表
    temp_1, temp_2 = zhan.pop()  # 弹出当前像素，赋值到temp_1, temp_2中
    a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2] # 提取当前像素周围的3x3邻域
    # 循环遍历3*3领域的每一个像素点，如果在高阈值与低阈值中间，就将对应区域变为255，存入
    if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 - 1] = 255
        zhan.append([temp_1 - 1, temp_2 - 1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        img_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
        img_yizhi[temp_1, temp_2 + 1] = 255
        zhan.append([temp_1, temp_2 + 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])

# 将非边缘的像素处理为0
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
plt.axis('off')  # 关闭坐标刻度值
plt.show()
