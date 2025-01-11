import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

img = plt.imread("lenna.png")
# 1 灰度化
img = np.dot(img, [0.3, 0.59, 0.11]) * 255
plt.imshow(img, cmap='gray')
plt.axis('off')

# 2 高斯平滑
# 2.1 计算高斯核
sigma = 0.5 # 高斯核标准差
dim = 5 # 高斯核大小
# [-2, -1, 0, 1, 2]
tmp = np.arange(-dim // 2 + 1, dim // 2 + 1)
x, y = np.meshgrid(tmp, tmp)
n1 = 1 / (2 * np.pi * sigma ** 2)
n2 = -1 / (2 * sigma ** 2)
Gaussian_filter = n1 * np.exp(n2 * (x ** 2 + y ** 2))
Gaussian_filter /= Gaussian_filter.sum() # 归一化
# 2.2 平滑操作
dx, dy = img.shape
padding_width = dim // 2
img_pad = np.pad(img, ((padding_width, padding_width), (padding_width, padding_width)), 'constant')
img_new = np.zeros((dx, dy))
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = (img_pad[i:i + dim, j:j + dim] * Gaussian_filter).sum()
plt.figure(2)
plt.imshow(img_new, cmap='gray')
# 3 边缘检测
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape)
img_tidu_y = np.zeros(img_new.shape)
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
        img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
img_tidu_x[img_tidu_x == 0] = 0.000000001
angle = img_tidu_y / img_tidu_x
plt.figure(3)
plt.imshow(img_tidu, cmap='gray')
# 4 非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx - 1):
    for j in range(1, dy - 1):
        tmp = img_tidu[i - 1: i + 2, j - 1: j + 2] # 8邻域
        flag = True # 标记是否抹除该点
        if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
            num_1 = (tmp[0, 1] - tmp[0, 0]) / angle[i, j] + tmp[0, 1]
            num_2 = (tmp[2, 1] - tmp[2, 2]) / angle[i, j] + tmp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] >= 1:
            num_1 = (tmp[0, 2] - tmp[0, 1]) / angle[i, j] + tmp[0, 1]
            num_2 = (tmp[2, 0] - tmp[2, 1]) / angle[i, j] + tmp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] > 0:
            num_1 = (tmp[0, 2] - tmp[1, 2]) * angle[i, j] + tmp[1, 2]
            num_2 = (tmp[2, 0] - tmp[1, 0]) * angle[i, j] + tmp[1, 0]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i, j] < 0:
            num_1 = (tmp[1, 0] - tmp[0, 0]) * angle[i, j] + tmp[1, 0]
            num_2 = (tmp[1, 2] - tmp[2, 2]) * angle[i, j] + tmp[1, 2]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i, j] = img_tidu[i, j]
plt.figure(4)
plt.imshow(img_yizhi, cmap='gray')

# 4 双阈值检测
low_boundary = img_tidu.mean() * 3
high_boundary = low_boundary * 3
zhan = []
for i in range(1, img_yizhi.shape[0] - 1):
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high_boundary:
            img_yizhi[i, j] = 255
            zhan.append((i, j))
        elif img_yizhi[i, j] <= low_boundary:
            img_yizhi[i, j] = 0

while len(zhan) != 0:
    edge_i, edge_j = zhan.pop()
    tmp = img_yizhi[edge_i - 1: edge_i + 2, edge_j - 1: edge_j + 2]
    for di, dj in [(-1, 1), (-1, 0), (-1, -1), (0, 1), (0, -1), (1, 1), (1, 0), (1, -1)]:
        if low_boundary < img_yizhi[edge_i + di, edge_j + dj] < high_boundary:
            zhan.append((edge_i + di, edge_j + dj))
            img_yizhi[edge_i + di, edge_j + dj] = 255
img_yizhi[(img_yizhi != 0) & (img_yizhi != 255)] = 0

plt.figure(5)
plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')

plt.show()
