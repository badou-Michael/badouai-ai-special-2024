# 随机生成符合高斯分布的随机数
import numpy as np
import cv2
from numpy import shape
import random
# 高斯噪声，高斯噪声指它的概率密度函数服从高斯分布（即正态分布）的一类噪声。
def GsNoise1(src, means, sigma, percet):
    NoiseImage = src
    NosieNum = int(percet*src.shape[0]*src.shape[1]) # percet指要添加噪声的范围
    for i in range(NosieNum):
        randX = random.randint(0,src.shape[0]-1)  # 随机生成的行
        randY = random.randint(0,src.shape[1]-1)  # 随机生成的列
        NoiseImage[randX, randY]=NoiseImage[randX, randY] + random.gauss(means, sigma) # 原像素灰度值加上随机数
        if NoiseImage[randX, randY] < 0:  # 灰度值小于0，取值为0
            NoiseImage[randX, randY] = 0
        elif NoiseImage[randX, randY] > 255:  # 灰度值大于255取值为255
            NoiseImage[randX, randY] = 255
    return NoiseImage

# 椒盐噪声
# 椒盐噪声也称为脉冲噪声，是一种随机出现的白点或者黑点
def JyNoise(image, pert):
    NoiseImage = image
    NoiseNum = int(pert*image.shape[0]*image.shape[1]) # pert指要添加噪声的范围
    for i in range(NoiseNum):
        randx = random.randint(0, image.shape[0] - 1)  # 随机生成的行
        randy= random.randint(0, image.shape[1] - 1)  # 随机生成的列
        # random.random()生成随机浮点数，一半概率是255，一半概率是0
        if random.random() <= 0.5:
            NoiseImage[randx, randy] = 0
        else:
            NoiseImage[randx, randy] = 255
    return NoiseImage


img = cv2.imread('lenna.png', 0)
img1 = GsNoise1(img, 2, 4, 0.7)  # 调用设置的高斯噪声函数
img2 = cv2.imread('lenna.png')
img3 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# 画图
cv2.imshow('source', img3)
cv2.imshow('lenna_GsNoise', img1)  # 加高斯噪声图

imgjiaoyan = JyNoise(img, 0.75)   # 调用设置的椒盐噪声函数
cv2.imshow('lenna_jiaoyan', imgjiaoyan)  # 椒盐噪声图
cv2.waitKey(0)