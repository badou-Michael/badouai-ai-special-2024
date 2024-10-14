# 椒盐噪声：随机出现的白点或者黑点
# 椒噪声（pepper noise,0）+ 盐噪声（salt noise,255）
import cv2
import random
import numpy
from numpy import shape
# src:图片； percentage:要加噪声的像素百分比
# 定义椒盐噪声函数：输入参数
def Pep_Salt(src,percentage):
    NoiseImg = src
    # 计算加噪音的像素点数
    NoiseNum = int(src.shape[0]*src.shape[1]*percentage)
    for i in range(NoiseNum):
        #随机选取要加噪音的像素点，randX为行，randY为Y
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
       # 生成随机数，小于0.5时，噪音设为0；大于0.5时，噪音设为255
        if random.random() <= 0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY] = 255
    return NoiseImg

# 读取一张灰度图片
img = cv2.imread('lenna.png',0)
# 返回添加椒盐噪声的图像
img1 = Pep_Salt(img,0.3)

# 读取一张彩色图片
img = cv2.imread('lenna.png')
# # 将 彩色图片转换为灰度图片
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('source1',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)         
