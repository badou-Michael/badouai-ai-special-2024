'''
yzc
canny 算法
1.灰度化
2.高斯滤波、高斯平滑
3.求梯度，sobel算子
4.非极大值抑制
5.双阈值检测
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def ImgGray(img,imgName):
    if imgName[-4:] == '.png':
        img = img * 255
    img_gray = img.mean(axis=-1)
    plt.figure(0)
    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    return img_gray
def Gauss(img,sigma,dim):
    Gauss_filter = np.zeros([dim,dim])#高斯核存储，数组
    tmp = [i-dim//2 for i in range(dim)] ##生成一个序列[-2,-1,0,1,2],以中心点坐标为原点
    print(tmp,'aaaa\n',Gauss_filter)
    # 计算高斯核
    n1 = 1/(2* math.pi * sigma**2)
    n2 = -1/(2 * sigma**2 )
    for i in range(dim):
        for j in range(dim):
            Gauss_filter[i,j] = n1 * math.exp(n2 * (tmp[i]**2 +tmp[j]**2))
    print('Gauss_filter\n', Gauss_filter, '222\n',Gauss_filter.sum())
    Gauss_filter = Gauss_filter / Gauss_filter.sum()

    dx,dy = img.shape
    img_new = np.zeros(img.shape)
    tmp = dim //2
    img_pad = np.pad(img,((tmp,tmp),(tmp,tmp)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i,j] =np.sum(img_pad[i:i+dim,j:j+dim] * Gauss_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8),cmap='gray')
    plt.axis('off')
    # plt.show()
    return img_new

def tidu(img):
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    dx,dy = img.shape
    img_tidu_x = np.zeros([dx,dy])
    img_tidu_y = np.zeros([dx,dy])
    img_tidu = np.zeros([dx,dy])
    img_pad = np.pad(img,((1,1),(1,1)),'constant')
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_x)
            img_tidu_y[i,j] = np.sum(img_pad[i:i+3,j:j+3]*sobel_kernel_y)
            img_tidu[i,j] = np.sqrt(img_tidu_x[i,j]**2 + img_tidu_y[i,j]**2)  #梯度大小的平方 = tidu_x **2 + tidu_y **2
    # if img_tidu_x == 0:
    #     img_tidu_x = 0.0000000001
    img_tidu_x[img_tidu_x ==0] =0.000000001  ##等同于62、63行
    angle = img_tidu_y / img_tidu_x
    plt.figure(2)
    plt.imshow(img_tidu.astype(np.uint8),cmap='gray')
    plt.axis('off')
    return img_tidu,angle

def yizhi(img_tidu,angle):
    img_yizhi = np.zeros(img_tidu.shape)
    dx, dy = img_tidu.shape
    for i in range(1,dx -1):
        for j in range(1,dy-1):
            flag = True
            temp = img_tidu[i-1:i+2,j-1:j+2] #8邻域矩阵，x轴坐标(i-1  i i+1 ) np.array([[0,0],[0,1],[0,1]])
            if angle[i,j] <= -1:
                num1 = (temp[0,1] - temp[0,0])/angle[i,j] +temp[0,1]
                num2 = (temp[2,1] - temp[2,2])/angle[i,j] +temp[2,1]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] >num2):
                    flag = False
            elif angle[i,j] <= 1:
                num1 = (temp[0,2] - temp[0,1])/angle[i,j] +temp[0,1]
                num2 = (temp[2,0] - temp[2,1])/angle[i,j] +temp[2,1]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] >num2):
                    flag = False
            elif angle[i,j] > 0:
                num1 = (temp[0,2] - temp[1,2])/angle[i,j] +temp[1,2]
                num2 = (temp[2,0] - temp[1,0])/angle[i,j] +temp[1,0]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] >num2):
                    flag = False
            elif angle[i,j] < 0:
                num1 = (temp[1,0] - temp[0,0])/angle[i,j] +temp[1,0]
                num2 = (temp[1,2] - temp[2,2])/angle[i,j] +temp[1,2]
                if not (img_tidu[i,j] > num1 and img_tidu[i,j] >num2):
                    flag = False
            if flag:
                img_yizhi[i,j] =img_tidu[i,j]
    return img_yizhi
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
    plt.axis('off')

def boundary(img,img_tidu):
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 不考虑外圈
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 将这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
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
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8),cmap='gray')
    plt.axis('off')



if __name__ == '__main__':
    imgName = 'lenna.png'
    # img = cv2.imread("..\\%s" %imgName)
    img = plt.imread("..\\%s" %imgName)
    img_gray = ImgGray(img,imgName)  #灰度化
    imgGs = Gauss(img_gray,0.5,5)  #高斯平滑
    img_tidu,angle = tidu(imgGs)  #求梯度
    img_yizhi = yizhi(img_tidu,angle)  #非极大值抑制
    boundary(img_yizhi,img_tidu)   #双阈值检测
    plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

