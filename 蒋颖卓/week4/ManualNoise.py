#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import random


def Gaussian_Noise(img, means, sigma, percentage):
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    # 循环所有像素
    for i in range(NoiseNum):
        # 随机选取一个像素点
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        # 输入参数sigma和mean,生成高斯随机数
        NoiseImg[randX, randY] = NoiseImg[randX, randY]+random.gauss(means, sigma)
    NoiseImg = np.clip(NoiseImg, 0, 255)

    # 输出图像
    return NoiseImg.astype(np.uint8)

def Salt_Pepper_Noise(img, percentage):
    NoiseImg = img.copy()
    NoiseNum = int(percentage * img.shape[0] * img.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


    # 输出图像
    return NoiseImg.astype(np.uint8)






img = cv2.imread('C:/Users/DMR/Desktop/1.png')
dst = Gaussian_Noise(img, 1, 50, 0.6)
dst1=Salt_Pepper_Noise(img,0.6)

cv2.imshow('Gaussian_Noise', dst)
cv2.imshow('Salt_Pepper_Noise', dst1)
cv2.waitKey()
