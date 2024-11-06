import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


if __name__== '__main__':

    image=cv2.imread('lenna.png')
    #1 灰度化 减少计算量
    image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #print(image_gray.shape)

    #2 高斯过滤 去噪

    sigma=2
    dim=5
    Gaussi_filter=np.zeros([dim,dim]) #卷积核的大小为5*5
    array=[i-dim//2 for i in range(dim)]   #构造一个序列
    #print(array)
    ##生成符合高斯分布的数的公式
    n1=1/(2*math.pi*sigma*sigma)
    n2=-1/(2*sigma*sigma)

    ##利用array的数打造一个满足5*5的，符合高斯分布的高斯核
    for i in range(dim):
        for j in range(dim):
            Gaussi_filter[i,j]=n1*math.exp(n2*(array[i]*array[i]+array[j]*array[j]))
    Gaussi_filter=Gaussi_filter/Gaussi_filter.sum()
    #print(Gaussi_filter)
    image_gaosi=np.zeros((image_gray.shape))#zeros((n,m))返回一个给定形状（n*m）和类型的用0填充的数组 存储经过高斯平滑后的图像
    pad=dim//2
    image_pad = np.pad(image_gray, ((pad, pad), (pad, pad)), 'constant')  # 边缘填补 在边缘扩充2圈
    #print(image_pad.shape)
    # np.pad()数组填充
    # ((1, 2), (3, 4)) 分别表示对行扩充(1,2),对列扩充(3,4)，顺序是先对行扩充，再对列扩充。1-在第一行之前扩充一行，2-在最后一行之后扩充2行
    # 'constant'表示连续填充相同的值
    # print('image_pad',image_pad.shape)

    dx,dy=image_gray.shape
    for i in range(dx):
        for j in range(dy):
            image_gaosi[i,j]=np.sum(image_pad[i:i+dim,j:j+dim]*Gaussi_filter)
    plt.figure(1)
    plt.imshow(image_gaosi.astype(np.uint8),cmap='gray')
    # np.astype() 强制类型转换
    # np.uint8() 将数据类型转换为无符号8位整数（即uint8），也就是整数范围在0到255之间。这个函数通常用于将像素值转换为uint8类型，以便在图像处理中进行操作和显示
    plt.axis('off')
    plt.show()#展示图片

    #3 Sobel算子（检测图像中的水平、垂直和对角边缘）
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # 检测垂直边缘
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # 检测水平边缘
    image_tidu_x=np.zeros((image_gaosi.shape))
    image_tidu_y=np.zeros((image_gaosi.shape))
    image_tidu=np.zeros((image_gaosi.shape))
    image_gaosi_pad=np.pad(image_gaosi,((1,1),(1,1)),'constant') #给经过高斯平滑的矩阵进行边缘填补
    for i in range(dx):
        for j in range(dy):
            image_tidu_x[i,j]=np.sum(image_gaosi_pad[i:i+3,j:j+3]*sobel_kernel_x)# x方向向右 因为卷积核的大小是3*3
            image_tidu_y[i,j]=np.sum(image_gaosi_pad[i:i+3,j:j+3]*sobel_kernel_y)# y方向向上
            image_tidu[i,j]=np.sqrt(image_tidu_x[i,j]*image_tidu_x[i,j]+image_tidu_y[i,j]*image_tidu_y[i,j])# 同时做X方向，y方向的卷积 np.sqrt() 开根号
    image_tidu_x[image_tidu_x==0]=0.00000001
    angle=image_tidu_y/image_tidu_x   #梯度的斜率 分母不为0
    plt.figure(2)
    plt.imshow(image_tidu.astype(np.uint8),cmap='gray')
    plt.axis('off')
    plt.show()

    #4 非极大值抑制
    image_yizhi=np.zeros((image_tidu.shape))
    for i in range(1,dx-1):
        for j in range(1,dx-1):
            flag=True
            temp=image_tidu[i-1:i+3,j-1:j+3] #[i,j]点的周围8邻域
            if angle[i,j]>=1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (image_tidu[i,j]>num_1 and image_tidu[i,j]>num_2):
                    flag=False
            elif angle[i,j]<=-1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 0]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 2]
                if not (image_tidu[i, j] > num_1 and image_tidu[i, j] > num_2):
                    flag = False
            elif angle[i,j]>0:
                num_1 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                if not (image_tidu[i, j] > num_1 and image_tidu[i, j] > num_2):
                    flag = False
            elif angle[i,j]<0:
                num_1 = (temp[1, 0] - temp[0, 0]) / angle[i, j] + temp[0, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) / angle[i, j] + temp[2, 2]
                if not (image_tidu[i, j] > num_1 and image_tidu[i, j] > num_2):
                    flag = False
            if flag:
                image_yizhi[i,j]=image_tidu[i,j]
    plt.figure(3)
    plt.imshow(image_yizhi.astype(np.uint8),cmap='gray')
    plt.axis('off')
    plt.show()

    #5 双阈值算法检测和连接边缘
    low_yuzhi=image_tidu.mean()*2
    high_yuzhi=low_yuzhi*2000
    zhan=[]
    for i in range(dx-1):
        for j in range(dy-1):
            if image_yizhi[i,j]>=high_yuzhi:
                image_yizhi[i,j]=255
                zhan.append([i,j])
            if image_yizhi[i,j]<=low_yuzhi:
                image_yizhi[i,j]=0

    while not len(zhan)==0: #如果列表zhan的长度不为0的话
        n1,n2=zhan.pop() # 出栈 pop() 删除并返回一个元素（默认为最后一个元素）
        #确定一个8邻域的范围
        temp1=image_yizhi[n1-1:n1+2,n2-1:n2+2]
        if (temp1[0,0]>low_yuzhi and temp1[0,0]<high_yuzhi):#因为已经确定了[n1,n2]是强边缘，所以[n1,n2]的8邻域的若边缘点可以确定为其不为噪点，所以可以纳入边缘
            temp1[0,0]=255
            zhan.append([n1-1,n2-1])
        if (temp1[0,1]>low_yuzhi and temp1[0,1]<high_yuzhi):
            temp1[0,1]=255
            zhan.append([n1 - 1, n2 ])
        if (temp1[0,2]>low_yuzhi and temp1[0,2]<high_yuzhi):
            temp1[0,2]=255
            zhan.append([n1 - 1, n2 + 1])
        if (temp1[1,0]>low_yuzhi and temp1[1,0]<high_yuzhi):
            temp1[1,0]=255
            zhan.append([n1 , n2 - 1])
        if (temp1[1,2]>low_yuzhi and temp1[1,2]<high_yuzhi):
            temp1[1,2]=255
            zhan.append([n1 , n2 + 1])
        if (temp1[2,0]>low_yuzhi and temp1[2,0]<high_yuzhi):
            temp1[2,0]=255
            zhan.append([n1 + 1, n2 - 1])
        if (temp1[2,1]>low_yuzhi and temp1[2,1]<high_yuzhi):
            temp1[2,1]=255
            zhan.append([n1 + 1, n2 ])
        if (temp1[2,2]>low_yuzhi and temp1[2,2]<high_yuzhi):
            temp1[2,2]=255
            zhan.append([n1 + 1, n2 + 1])

    for i in range(image_yizhi.shape[0]):
        for j in range(image_yizhi.shape[1]):
            if image_yizhi[i,j] != 0 and image_yizhi[i,j] != 255:
                image_yizhi[i,j]==0

    plt.figure(4)
    plt.imshow(image_yizhi.astype(np.uint8),cmap='gray')
    plt.axis('off')
    plt.show()
