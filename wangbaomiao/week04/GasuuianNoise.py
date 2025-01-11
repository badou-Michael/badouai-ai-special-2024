# -*- coding: utf-8 -*-
# time: 2024/10/17 16:48
# file: GasuuianNoise.py
# author: flame
import numpy as np
import cv2
from numpy import shape
import random

# 随机生成符合正态高斯分布的随机数，means，sigma两个参数
def GasuianNoise(src,means,sigma,percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        NoiseImg[randx,randy] = NoiseImg[randx,randy] + random.gauss(means,sigma)
        if NoiseImg[randx,randy] < 0:
            NoiseImg[randx,randy] = 0
        elif NoiseImg[randx,randy] > 255:
            NoiseImg[randx,randy] = 255
    return NoiseImg

img = cv2.imread("lenna.png",0)
img1 = GasuianNoise(img,2,4,0.8)
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("src",img2)
cv2.imshow("dst",img1)
cv2.imshow("1_2",np.hstack([img2,img1]))
cv2.waitKey()