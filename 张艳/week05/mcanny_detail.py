import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# 0、彩色图转灰度图，均值法
name = 'lenna.png'
img = plt.imread(name)
if name[-4:] == '.png':
    img = img * 255  # 0.0~1.0 --> 0.0~255.0
img = img.mean(axis=-1)  # w*h*3 --> w*h*1

# 1、高斯平滑
# 高斯核
size = 5 #卷积核大小
half = size//2
sigma = 0.5 #卷积核参数
n1 = 1 / (2 * np.pi * sigma ** 2)
n2 = -1 / (2 * sigma ** 2)
arr = [(i - half) for i in range(size)]  # [-2,-1,0,-1,-2]
guass_filter = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        guass_filter[i, j] = n1 * math.exp(n2 * (arr[i] ** 2 + arr[j] ** 2))
guass_filter = guass_filter / guass_filter.sum()
# 高斯滤波
img_pad = np.pad(img, pad_width=((half, half), (half, half)), mode='constant', constant_values=0)
img_1 = np.zeros(img.shape)
w, h = img.shape # err: w = img.shape[0], h = img.shape[1],,必须分两行写
for i in range(w):
    for j in range(h):
        img_1[i,j] = np.sum(img_pad[i:i+size,j:j+size] * guass_filter) # err: img_pad[i:i+size][j:j+size]
plt.figure(1)
plt.imshow(img_1.astype(np.uint8),cmap='gray')
plt.title('img_1')

# 2、求梯度
sobel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
sobel_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
img_2_gradx = np.zeros(img_1.shape)
img_2_grady = np.zeros(img_1.shape)
img_2_grad = np.zeros(img_1.shape)
img_1_pad = np.pad(img_1, pad_width=((1,1),(1,1)), mode='constant')
for i in range(w):
    for j in range(h):
        img_2_gradx[i,j] = np.sum(img_1_pad[i:i+3,j:j+3] * sobel_x)
        img_2_grady[i,j] = np.sum(img_1_pad[i:i+3,j:j+3] * sobel_y)
        img_2_grad[i,j] = np.sqrt(img_2_gradx[i,j]**2 + img_2_grady[i,j]**2)
plt.figure(2)
plt.imshow(img_2_grad.astype(np.uint8),cmap='gray')
plt.title('img_2_grad')

# 3、非极大值抑制
img_2_gradx[img_2_gradx==0] = 0.00000001
tan = img_2_grady / img_2_gradx
img_3 = np.zeros(img_2_grad.shape)
for i in range(1,w-1):
    for j in range(1,h-1):
        grad_temp = img_2_grad[i-1:i+2,j-1:j+2]
        flag = True
        if tan[i,j]<=-1:
            num_1 = (grad_temp[0,1] - grad_temp[0,0])/tan[i,j] + grad_temp[0,1]
            num_2 = (grad_temp[2,1] - grad_temp[2,2])/tan[i,j] + grad_temp[2,1]
            if not (grad_temp[1,1]>num_1 and grad_temp[1,1]>num_2):
                flag = False
        elif tan[i,j]>=1:
            num_1 = (grad_temp[0,2] - grad_temp[0,1])/tan[i,j] + grad_temp[0,1]
            num_2 = (grad_temp[2,0] - grad_temp[2,2])/tan[i,j] + grad_temp[2,1]
            if not (grad_temp[1,1]>num_1 and grad_temp[1,1]>num_2):
                flag = False
        elif tan[i,j]>0:
            num_1 = (grad_temp[0,2] - grad_temp[1,2])*tan[i,j] + grad_temp[1,2]
            num_2 = (grad_temp[2,0] - grad_temp[1,0])*tan[i,j] + grad_temp[1,0]
            if not (grad_temp[1,1]>num_1 and grad_temp[1,1]>num_2):
                flag = False
        elif tan[i,j]<0:
            num_1 = (grad_temp[1,0] - grad_temp[0,0])*tan[i,j] + grad_temp[1,0]
            num_2 = (grad_temp[1,2] - grad_temp[2,2])*tan[i,j] + grad_temp[1,2]
            if not (grad_temp[1,1]>num_1 and grad_temp[1,1]>num_2):
                flag = False
        if flag:
            img_3[i,j] = grad_temp[1,1]
plt.figure(3)
plt.imshow(img_3.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.title('img_3')

# 4、双阈值检测，连接边缘。
low_threshold = img_2_grad.mean() * 0.5  # img_3.mean() / 2 # ?
high_threshold = low_threshold * 3
zhan = [] # 栈
#强边缘、非边缘
for i in range(1,w-1):
    for j in range(1,h-1):
        if img_3[i,j]>high_threshold :
            img_3[i,j] = 255
            zhan.append([i,j]) # 语法
        elif img_3[i,j]<low_threshold :
            img_3[i,j] = 0
#弱边缘-里的非孤立点
while len(zhan)>0:
    x,y = zhan.pop()
    arr = img_3[x-1:x+2, y-1:y+2]
    if (arr[0,0]>low_threshold and arr[0,0]<high_threshold):
        img_3[x-1,y-1] = 255
        zhan.append([x-1,y-1])
    if (arr[0,1]>low_threshold and arr[0,1]<high_threshold):
        img_3[x-1,y] = 255
        zhan.append([x-1,y])
    if (arr[0,2]>low_threshold and arr[0,2]<high_threshold):
        img_3[x-1,y+1] = 255
        zhan.append([x-1,y+1])
    if (arr[1,0]>low_threshold and arr[1,0]<high_threshold):
        img_3[x,y-1] = 255
        zhan.append([x,y-1])
    if (arr[1,2]>low_threshold and arr[1,2]<high_threshold):
        img_3[x,y+1] = 255
        zhan.append([x,y+1])
    if (arr[2,0]>low_threshold and arr[2,0]<high_threshold):
        img_3[x+1,y-1] = 255
        zhan.append([x+1,y-1])
    if (arr[2,1]>low_threshold and arr[2,1]<high_threshold):
        img_3[x+1,y] = 255
        zhan.append([x+1,y])
    if (arr[2,2]>low_threshold and arr[2,2]<high_threshold):
        img_3[x+1,y+1] = 255
        zhan.append([x+1,y+1])
#弱边缘-里的孤立点
for i in range(w):
    for j in range(h):
        if (img_3[i,j]!=0) and (img_3[i,j]!=255):
            img_3[i,j]=0
plt.figure(4)
plt.imshow(img_3.astype(np.uint8), cmap='gray')
plt.axis('off')
plt.title('img_4')
plt.show()
