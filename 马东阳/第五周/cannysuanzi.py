# canny边缘检测
# 高斯滤波，平滑图像，滤除噪声
# 计算图像中每个像素点的梯度强度和方向
# 应用非极大值抑制，消除边缘计算的杂散响应
# 双阈值检测确定真实和潜在的边缘
# 通过抑制孤立的弱边缘完成边缘检测

import numpy as np
import matplotlib.pyplot as plt
import math

img= plt.imread('lenna.png')
print("image", img)
pic_path = 'lenna.png'
if pic_path[-4:] == '.png':
    img = img * 255
img = img.mean(axis=-1)

# 高斯平滑
sigma = 0.6
dim = 5
Gass_filter = np.zeros([dim,dim])
tmp = [i-dim//2 for i in range(dim)]
n1 = 1/(2*math.pi*sigma**2)
n2 = -1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        Gass_filter[i,j] = n1 * math.exp(n2*(tmp[i]**2)+ tmp[j]**2)
Gass_filter = Gass_filter/Gass_filter.sum()
dx,dy = img.shape
img_new = np.zeros(img.shape)  # 存储平滑后的图像
tmp = dim//2
img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant') #边缘填补
for i in range(dx):
    for j in range(dy):
        img_new[i,j] = np.sum(img_pad[i:i+dim,j:j+dim]*Gass_filter)
plt.figure(1)
plt.imshow(img_new.astype(np.uint8),cmap='gray')
plt.axis('off')

# 求梯度 sobel矩阵
sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_kernel_y = np.array([[1,2,1],[0,0,0],[-1,2,1]])
img_tidu_x = np.zeros(img_new.shape)  #存梯度图像
img_tidu_y = np.zeros([dx,dy])
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new,((1,1),(1,1)),'constant') #边缘填补
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
        img_tidu_y[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
        img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2 + img_tidu_y[i,j]**2)
img_tidu_x[img_tidu_x == 0] = 0.00000001
angle = img_tidu_y/img_tidu_x
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8),cmap='gray')
plt.axis('off')

# 非极大值抑制
img_yizhi = np.zeros(img_tidu.shape)
for i in range(1, dx-1):
    for j in range(1, dy-1):
        flag = True #标记
        temp = img_tidu[i-1:i+2,j-1:j+2]
        if angle[i,j] <= -1:
            num_1 = (temp[0,1]-temp[0,0])/angle[i,j] + temp[0,1]
            num_2 = (temp[2,1]-temp[2,2])/angle[i,j] + temp[2,1]
            if not(img_tidu[i,j] > num_1 and img_tidu[i,j] > num_2):
                flag = False
        elif angle[i,j] >= 1:
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i,j] > 0:
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        elif angle[i,j] < 0:
            num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                flag = False
        if flag:
            img_yizhi[i,j] = img_tidu[i,j]
plt.figure(3)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')

# 双阈值检测
low_boundary = img_tidu.mean()*0.5
high_boundary = low_boundary*3
zhan = [] # 栈
for i in range(1,img_yizhi.shape[0]-1):
    for j in range(i,img_yizhi.shape[1]-1):
        if img_yizhi[i,j] >= high_boundary:
            img_yizhi[i,j]=255
            zhan.append([i,j])
        elif img_yizhi[i,j] <= low_boundary:
            img_yizhi[i,j] = 0
while not len(zhan) == 0:
    temp_1, temp_2 = zhan.pop() #出栈
    a = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if (a[0,0] < high_boundary) and (a[0,0] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
    if (a[0,1] < high_boundary) and (a[0,1] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
    if (a[0,2] < high_boundary) and (a[0,2] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
    if (a[1,0] < high_boundary) and (a[1,0] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
    if (a[1,2] < high_boundary) and (a[1,2] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
    if (a[2,0] < high_boundary) and (a[2,0] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
    if (a[2,1] < high_boundary) and (a[2,1] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
    if (a[2,2] < high_boundary) and (a[2,2] > low_boundary):
        img_yizhi[temp_1-1,temp_2-1] = 255
        zhan.append([temp_1-1,temp_2-1])
for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i,j] !=0 and img_yizhi[i,j] != 255:
            img_yizhi[i,j] = 0

#绘图
plt.figure(4)
plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')
plt.show()

# 方式2 通过cv接口实现
import cv2
img1 = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)
edg = cv2.Canny(img1,100,200)

cv2.imshow('canny edge detection', edg)
cv2.waitKey(0)
cv2.destroyAllWindows()
