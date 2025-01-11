"""
    增加椒盐噪声
"""
import numpy as np
import cv2
from numpy import shape
import random

# 定义函数，用于给图像添加椒盐噪声
def JiaoYanNoise(src, percetage):
    # 将输入图像赋值给NoiseImg，作为添加噪声后的图像
    NoiseImg = src
    # 根据输入图像的尺寸和给定的噪声比例，计算要添加的噪声点数量
    NoiseNum = int(percetage * src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        # 随机生成噪声点的横坐标，范围在0到图像行数减1之间
        randX = random.randint(0,src.shape[0]-1)
        # 随机生成噪声点的纵坐标，范围在0到图像列数减1之间
        randY = random.randint(0,src.shape[1]-1)
        # 在随机位置设置为0或255，模拟椒盐噪声
        NoiseImg[randX, randY] = random.choice([0, 255])
    # 返回添加噪声后的图像
    return NoiseImg

# 读取灰度图像
img = cv2.imread('lenna.png', 0)
# 对灰度图像添加椒盐噪声，噪声比例为0.4
img1 = JiaoYanNoise(img, 0.4)
# 读取彩色图像
img = cv2.imread('lenna.png')
# 将彩色图像转换为灰度图像
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示原始彩色图像
cv2.imshow('lenna_color', img)
# 显示添加椒盐噪声后的灰度图像
cv2.imshow('lenna_JiaoYanNoise', img1)
# 显示原始彩色图像转换后的灰度图像
cv2.imshow('lenna_grey', img2)
# 等待用户按键，参数0表示无限等待
cv2.waitKey(0)
