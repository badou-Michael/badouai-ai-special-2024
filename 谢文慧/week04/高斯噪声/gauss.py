# 高斯噪声，给一副图像加上高斯噪声顺序
import cv2
import numpy as np
from numpy import shape
import random
# src:图片；-   means和sigma:高斯分布参数；   percentage:要加噪声的像素百分比
# 1、输入参数sigma、和mean
def GaussianNoise(src,means,sigma,percentage):
    NoiseImg = src
    print(src.shape,src.shape[0],src.shape[1])
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        # 每次随机取一个点
        # 把一张图片的像素用行和列表示，randX表示随机生成的行值，randY表示随机生成的列值
        randX = random.randint(0,src.shape[0]-1) #左右均包含在随机范围内
        randY = random.randint(0,src.shape[1]-1)
       
        # 2、生成高斯随机数
        gauss_random=random.gauss(means,sigma)
        
        # 3、根据输入像素计算出输出像素,在原有像素灰度值上加随机数
        NoiseImg[randX,randY] = NoiseImg[randX,randY]+gauss_random
        
        # 4、重新将像素值放缩在【0-255】之间
        # 若灰度值小于0,则强制为0；若灰度值大于255,则强制为255
        if NoiseImg[randX,randY] < 0:
            NoiseImg[randX,randY] = 0
        elif NoiseImg[randX,randY] >255:
            NoiseImg[randX,randY] = 255
        # 5、循环所有像素
    
    # 6、输出图像
    return NoiseImg

# cv2.imread(filename, flags)
# 参数：
# filepath：读入img的完整路径
# flags：标志位，{cv2.IMREAD_COLOR，cv2.IMREAD_GRAYSCALE，cv2.IMREAD_UNCHANGED}
# cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，可用1作为实参替代
# cv2.IMREAD_GRAYSCALE：读入灰度图片，可用0作为实参替代
# cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道，可用-1作为实参替代
# 读取一张灰度图片
img = cv2.imread('lenna.png',0)
# print(img)
# print("------")
cv2.imshow('source',img)
# 返回添加高斯噪声的图像
img1 = GaussianNoise(img,2,3,0.8)
# 读取一张彩色图片
# img = cv2.imread('lenna.png')
# # 将 彩色图片转换为灰度图片
# img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('source1',img)
# print(img)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)
