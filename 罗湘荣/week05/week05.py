import math
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#图像灰度化：
photo = cv2.imread("ho.jpg", 0)
cv2.imshow("灰度化",photo)
#1、高斯平滑
#sigma=1.52 #高斯平滑时的高斯核参数，标准差，可调
sigma=0.5 #高斯平滑时的高斯核参数，标准差，可调
dim=5 #高斯核尺寸
Gaussian_filter = np.zeros([dim , dim])#存储高斯核，数组
tmp=[i-dim//2 for i in range(dim)] #生成一个序列
n1=1/(2*math.pi*sigma**2)
n2=-1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        Gaussian_filter[i,j]=n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
Gaussian_filter=Gaussian_filter/Gaussian_filter.sum()
dx,dy=photo.shape
#存储平滑之后的图像
photo_new =np.zeros(photo.shape)
tmp=dim//2
#边缘填补
photo_pad=np.pad(photo,((tmp,tmp),(tmp,tmp)),'constant')
for i in range(dx):
    for j in range(dy):
        photo_new[i,j]=np.sum(photo_pad[i:i+dim,j:j+dim]*Gaussian_filter)
plt.figure(1)
plt.imshow(photo_new.astype(np.uint8),cmap='gray')
plt.axis('off')

#2、求梯度：
sobel_kernel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_kernel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
#存储梯度图像
photo_tidu_x= np.zeros(photo_new.shape)
photo_tidu_y=np.zeros([dx,dy])
photo_tidu=np.zeros(photo_new.shape)
#边缘填补，根据上面矩阵结构所以写1？
photo_pad=np.pad(photo_new,((1,1),(1,1)),'constant')
for i in range(dx):
    for j in range(dy):
        photo_tidu_x[i,j]=np.sum(photo_pad[i:i+3,j:j+3]*sobel_kernel_x)
        photo_tidu_y[i,j]=np.sum(photo_pad[i:i+3,j:j+3]*sobel_kernel_y)
        photo_tidu[i,j]=np.sqrt(photo_tidu_x[i,j]**2+photo_tidu_y[i,j]**2)
photo_tidu_x[photo_tidu_x==0]=0.000000001
angle = photo_tidu_y/photo_tidu_x
plt.figure(2)
plt.imshow(photo_tidu.astype(np.uint8),cmap='gray')
plt.axis('off')

#3、非极大值抑制
photo_yizhi =np.zeros(photo_tidu.shape)
for i in range(1,dx-1):
    for j in range(1,dy-1):
        flag=True
        temp=photo_tidu[i-1:i+2 , j-1:j+2]
        if angle[i,j]<= -1:
            num_1=(temp[0,1]-temp[0,0])/angle[i,j]+temp[0,1]
            num_2=(temp[2,1]-temp[2,2])/angle[i,j]+temp[2,1]
            if not (photo_tidu[i,j]>num_1 and photo_tidu[i,j]>num_2):
                flag= False
        elif angle[i,j]>=1:
            num_1=(temp[0,2]-temp[0,1])/angle[i,j]+temp[0,1]
            num_2=(temp[2,0]-temp[2,1])/angle[i,j]+temp[2,1]
            if not (photo_tidu[i,j]>num_1 and photo_tidu[i,j]>num_2):
                flag= False
        elif angle[i,j]>0:
            num_1=(temp[0,2]-temp[0,1])*angle[i,j]+temp[1,2]
            num_2=(temp[2,0]-temp[1,0])*angle[i,j]+temp[1,0]
            if not (photo_tidu[i,j]>num_1 and photo_tidu[i,j]>num_2):
                flag= False
        elif angle[i,j]<0:
            num_1=(temp[1,0]-temp[0,0])*angle[i,j]+temp[1,0]
            num_2=(temp[1,2]-temp[2,2])*angle[i,j]+temp[1,2]
            if not (photo_tidu[i,j]>num_1 and photo_tidu[i,j]>num_2):
                flag= False
            if flag:
                photo_yizhi[i,j]=photo_tidu[i,j]
plt.figure(3)
plt.imshow(photo_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')
#4、双阀值检测，连接边缘。
lower_boundary =photo_tidu.mean()*0.5
high_boundary = lower_boundary*3
zhan=[]
for i in range(1,photo_yizhi.shape[0]-1):
    for j in range(1,photo_yizhi.shape[1]-1):
        if photo_yizhi[i,j]>=high_boundary:
            photo_yizhi[i,j]=255
            zhan.append([i,j])
        elif photo_yizhi[i,j] <= lower_boundary:
            photo_yizhi[i,j] = 0
while not len(zhan)==0:
    temp_1,temp_2=zhan.pop()
    a = photo_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if(a[0,0]<high_boundary) and (a[0,0]>lower_boundary):
        photo_yizhi[temp_1-1,temp_2-1]=255
        zhan.append([temp_1-1,temp_2-1])
    if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
        photo_yizhi[temp_1 - 1, temp_2] = 255
        zhan.append([temp_1 - 1, temp_2])
    if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
        photo_yizhi[temp_1 - 1, temp_2 + 1] = 255
        zhan.append([temp_1 - 1, temp_2 + 1])
    if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
        photo_yizhi[temp_1, temp_2 - 1] = 255
        zhan.append([temp_1, temp_2 - 1])
    if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
        photo_yizhi[temp_1 + 1, temp_2 - 1] = 255
        zhan.append([temp_1 + 1, temp_2 - 1])
    if (a[2,1] < high_boundary) and (a[2,1] > lower_boundary):
        photo_yizhi[temp_1 +1, temp_2] = 255
        zhan.append([temp_1 + 1, temp_2])
    if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
        photo_yizhi[temp_1 + 1, temp_2 + 1] = 255
        zhan.append([temp_1 + 1, temp_2 + 1])
for i in range(photo_yizhi.shape[0]):
    for j in range(photo_yizhi.shape[1]):
        if photo_yizhi[i,j]!=0 and photo_yizhi[i,j]!=255:
            photo_yizhi[i,j]=0

plt.figure(4)
plt.imshow(photo_yizhi.astype(np.uint8),cmap='gray')
plt.axis('off')
plt.show()
