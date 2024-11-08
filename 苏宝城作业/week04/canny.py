import cv2 as cv
import matplotlib.pyplot as plt
import math
import numpy as np

img = cv.imread('lenna.png', 0)

#高斯平滑 过程：建立高斯核 通过高斯核函数对原图进行卷积
simga = 0.5  #高斯核参数
dim = 5  #核尺寸
Gaussian_filter = np.zeros([dim, dim])  #存储高斯核（数组）
tmp = [i - dim // 2 for i in range(dim)]
n1 = 1/(2*math.pi*sigma**2) #计算高斯核
n2 = -1/(2*sigma**2)
for i in range(dim):
  for j in range(dim):
    Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
dx, dy = img.shape
img_new = np.zeros(img.shape)  #存储平滑之后的图像，zeros函数得到的是浮点型数据
tmp = dim//2
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)

#求梯度
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
img_tidu_y = np.zeros([dx, dy])
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)  # x方向
        img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)  # y方向
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
img_tidu_x[img_tidu_x == 0] = 0.00000001
angle = img_tidu_y/img_tidu_x   #numpy矩阵 为每个像素点的梯度向量

#非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx-1):
  for j in range(1, dy-1):
    flag = True      #在8领域内是否要抹去做个标记
    temp = img_tidu[i-1:i+2, j-1:j+2]   #梯度幅值的8领域矩阵
    if angle[i, j] <= -1: #使用线性插值判断抑制与否
        num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
        num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
            flag = False
    elif angle[i, j] >= 1:
        num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
        num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
            flag = False
    elif angle[i, j] > 0:
        num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
        num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
          flag = False
    elif angle[i, j] < 0:
        num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
        num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
        if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
            flag = False
    if flag:
        img_yizhi[i, j] = img_tidu[i, j]

#双阈值检测，连接边缘去遍历一定是边的点。若其8领域内存在可能是边的点，则进栈
lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0
 
    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1-1, temp_2-1])  # 进栈
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
 
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
 
