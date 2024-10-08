
"""
    增加高斯噪声
"""
import numpy as np
import cv2
from numpy import shape
import random

# 定义函数，用于给图像添加高斯噪声（图像，均值，方差，百分比）
def GaoSiNoise(src, means, sigma, percetage):
    # 将输入图像赋值给NoiseImg，作为添加噪声后的图像
    NoiseImg = src
    # 根据输入图像的尺寸和给定的噪声比例，计算要添加的噪声点数量
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        # 随机生成噪声点的横坐标，范围在0到图像行数减1之间
        randX = random.randint(0, src.shape[0] - 1)
        # 随机生成噪声点的纵坐标，范围在0到图像列数减1之间
        randY = random.randint(0, src.shape[1] - 1)
        # 在随机位置添加具有给定均值和标准差的高斯噪声
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        # 如果噪声后的像素值小于0，将其设置为0
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        # 如果噪声后的像素值大于255，将其设置为255
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    # 返回添加噪声后的图像
    return NoiseImg

# 读取灰度图像
img = cv2.imread('lenna.png', 0)
# 对灰度图像添加高斯噪声，均值为2，标准差为4，噪声比例为0.9
#常见均值方差为2，4
img1 = GaoSiNoise(img, 2, 4, 0.9)
# 读取彩色图像
img = cv2.imread('lenna.png')
# 将彩色图像转换为灰度图像
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示添加噪声后的灰度图像和原始彩色图像转换后的灰度图像
cv2.imshow('lenna_color', img)
cv2.imshow('lenna_GaoSiNoise', img1)
cv2.imshow('lenna_grey', img2)
# 等待用户按键，参数0表示无限等待
cv2.waitKey(0)
