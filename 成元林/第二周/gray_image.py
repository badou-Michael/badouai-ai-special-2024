import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
# 读取图片
ori_img = cv2.imread("lenna.png")
#方法一：手动遍历循环
# 需要判断是否通道图像不是二值化图像
if ori_img.ndim>1:
    #转换为RGB通道
    ori_img = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    #灰度化过程，将3通道转为单通道
    # 需先创建一个初始化二维矩阵来接收单通道的数据
    #在创建二维矩阵之前，获取单通道的图像分辨率大小
    # g_shape = ori_img.shape
    # h,w = g_shape[0],g_shape[1]
    #创建一个新的二维矩阵
    # gray_img = np.zeros([h,w],dtype=ori_img.dtype)
    # # 循环遍历图像分辨率
    # for i in range(h):
    #     for j in range(w):
    #         #获取某个像素点的像素值,每个像素值都有rgb的像素点
    #         m = ori_img[i,j]
    #         #新的矩阵接收值，根据浮点算法得出 gray = R*0.3+G*0.59+B*0.11
    #         gray_img[i,j] = m[0]*0.3+m[1]*0.59+m[2]*0.11
    #可以用点积的方式，新的矩阵接收值，根据浮点算法得出 gray = R*0.3+G*0.59+B*0.11
    # print(ori_img[...,:3])
    gray_img = np.dot(ori_img[...,:3],[0.3,0.59,0.11])
    #创建画布
    plt.subplot(3,2,1)
    #绘制图像,cmap是显示成灰色
    plt.imshow(gray_img,cmap="gray")
    # print(gray_img)
    # 将灰度图像二值化
    # 设定阈值125
    threshold = 125
    binary_img = np.where(gray_img>threshold,255,0)
    # print(binary_img)
    plt.subplot(3,2,2)
    #绘制图像,cmap是显示成二值
    plt.imshow(binary_img,cmap="gray")
else:
    plt.imshow(ori_img,cmap="gray")
# plt.show()
# print('=====================使用OpenCV库==============================')
# 方法二：使用OpenCV库
ori_img1 = cv2.imread("lenna.png")
plt.subplot(3,2,3)
gray_img2 = cv2.cvtColor(ori_img1,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img2,cmap="gray")
plt.subplot(3,2,4)
binary_img1 = cv2.threshold(gray_img2, threshold, 255, cv2.THRESH_BINARY)[1]
plt.imshow(binary_img1,cmap="gray")
# plt.show()
# print('=====================使用sckimge库==============================')
ori_img2 = cv2.imread("lenna.png")
ori_img2 = cv2.cvtColor(ori_img1,cv2.COLOR_BGR2RGB)
gray_img3 = rgb2gray(ori_img2)
plt.subplot(3,2,5)
plt.imshow(gray_img3,cmap="gray")
#使用大津法找到图像的阈值
binary_img2 = np.where(gray_img3 >= 0.5, 1, 0)  # 是的话是1，否则为0
plt.subplot(3,2,6)
plt.imshow(binary_img2,cmap="gray")
plt.show()