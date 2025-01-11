"""
1. 灰度化
2. 高斯滤波
3. sobel算梯度
4. 非极大值抑制
5. 双阈值滤波
6. 滤除孤立点
"""

import numpy as np
import math
from matplotlib import pyplot as plt

img_path = "lenna.png"
# 1.灰度化
img_ori = plt.imread(img_path)
if img_path[-4:] == ".png":
    img_ori = img_ori * 255
img_ori = np.mean(img_ori, axis=-1)
    
# 2.高斯滤波
sigma = 0.5
dim = 5
center = dim // 2
gaus_filter = np.zeros((dim, dim))
n1 = 1/(2*math.pi*(sigma**2))
n2 = -1/(2*(sigma**2))
for i in range(dim):
    for j in range(dim):
        # 高斯核需要按中心对称
        gaus_filter[i,j] = n1*math.exp(((i-center)**2+(j-center)**2)*n2)
gaus_filter /= gaus_filter.sum()
img_gaus = np.zeros(img_ori.shape)
img_pad = np.pad(img_ori, ((2,2),(2,2)), "constant")
for i in range(img_gaus.shape[0]):
    for j in range(img_gaus.shape[1]):
        tmp = img_pad[i:i+5, j:j+5]
        img_gaus[i,j] = np.sum(tmp*gaus_filter)
plt.figure(1)
img_gaus = np.clip(img_gaus, 0, 255)
plt.imshow(img_gaus.astype(np.uint8), cmap="gray")
plt.axis("off")

# 3. 算梯度
sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
img_grad_x = np.zeros(img_gaus.shape)
img_grad_y = np.zeros(img_gaus.shape)
img_grad = np.zeros(img_gaus.shape)
img_pad = np.pad(img_gaus, ((1,1),(1,1)), "constant")
for i in range(img_gaus.shape[0]):
    for j in range(img_gaus.shape[1]):
        tmp = img_pad[i:i+3, j:j+3]
        img_grad_x[i,j] = np.sum(tmp*sobel_x)
        img_grad_y[i,j] = np.sum(tmp*sobel_y)
img_grad = np.sqrt(img_grad_x**2 + img_grad_y**2)
img_grad_x[img_grad_x==0] = 1e-8
img_angle = img_grad_y / img_grad_x

plt.figure(2)
img_grad = np.clip(img_grad, 0, 255)
plt.imshow(img_grad.astype(np.uint8), cmap="gray")
plt.axis("off")

# 4. 非极大值抑制
img_nms = np.zeros(img_grad.shape)
for i in range(1, img_grad.shape[0]-1):
    for j in range(1, img_grad.shape[1]-1):
        tmp = img_grad[i-1:i+2, j-1:j+2]
        angle = img_angle[i,j]
        # if -0.5 <= angle < 0.5:
        #     n1 = tmp[1,0]
        #     n2 = tmp[1,2]
        # elif 0.5 <= angle < 2:
        #     n1 = tmp[2,0]
        #     n2 = tmp[0,2]
        # elif abs(angle) >= 2:
        #     n1 = tmp[0,1]
        #     n2 = tmp[2,1]
        # elif -2 < angle < -0.5:
        #     n1 = tmp[0,0]
        #     n2 = tmp[2,2]
        if angle <= -1:
            n1 = (tmp[0,1]-tmp[0,0])/angle + tmp[0,1]
            n2 = (tmp[2,1]-tmp[2,2])/angle + tmp[2,1]
        elif angle >= 1:
            n1 = (tmp[0,2]-tmp[0,1])/angle + tmp[0,1]
            n2 = (tmp[2,0]-tmp[2,1])/angle + tmp[2,1]
        elif angle < 0:
            n1 = (tmp[1,0]-tmp[0,0])*angle + tmp[1,0]
            n2 = (tmp[1,2]-tmp[2,2])*angle + tmp[1,2]
        elif angle > 0:
            n1 = (tmp[2,0]-tmp[1,0])*angle + tmp[1,0]
            n2 = (tmp[0,2]-tmp[1,2])*angle + tmp[1,2]
        if img_grad[i,j] > n1 and img_grad[i,j] > n2:
            img_nms[i,j] = img_grad[i,j]
plt.figure(3)
img_nms = np.clip(img_nms, 0, 255)
plt.imshow(img_nms.astype(np.uint8), cmap="gray")
plt.axis("off")

# 双阈值滤波
# 滤波的阈值由img_grad提取，而不是img_nms
low_thd = img_grad.mean()*0.5
high_thd = low_thd * 3
print(high_thd, low_thd)

stack = []
img_thd = np.zeros(img_nms.shape)
for i in range(1, img_nms.shape[0]-1):
    for j in range(1, img_nms.shape[1]-1):
        if img_nms[i,j] >= high_thd:
            stack.append([i,j])
            img_thd[i,j] = 255
        elif img_nms[i,j] < low_thd:
            img_thd[i,j] = 0
        else:
            img_thd[i,j] = img_nms[i,j]
            
move = [[-1,-1],[-1,0],[-1,1],
        [0,-1],        [0,1],
        [1,-1], [1,0], [1,1]]
while len(stack) != 0:
    tmp = stack.pop()
    for dx,dy in move:
        if low_thd < img_thd[tmp[0]+dx, tmp[1]+dy] < high_thd:
            img_thd[tmp[0]+dx, tmp[1]+dy] = 255
            stack.append([tmp[0]+dx, tmp[1]+dy])

# 去除孤立点
for i in range(img_thd.shape[0]):
    for j in range(img_thd.shape[1]):
        if img_thd[i,j] != 255:
            img_thd[i,j] = 0
plt.figure(4)
plt.imshow(img_thd.astype(np.uint8), cmap="gray")
plt.axis("off")
plt.show()