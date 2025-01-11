# 高斯噪声

import numpy as np
import cv2
from numpy import shape
import random
def gaosi (src,means,sigma,percetage):
    NoiseImg = src
    NoiseSum = int(percetage * src.shape[0] * src.shape[1])
    for i in range (NoiseSum):
        tanX = random.randint(0,src.shape[0]-1)
        tanY = random.randint(0,src.shape[1]-1)

        NoiseImg[tanX,tanY] = NoiseImg[tanX,tanY] + random.gauss(means,sigma)

        if  NoiseImg[tanX,tanY] < 0:
            NoiseImg[tanX,tanY] = 0
        elif NoiseImg[tanX,tanY] >255:
             NoiseImg[tanX,tanY] = 255

    return NoiseImg

img = cv2.imread('lenna.png' , 0)
img1 = gaosi (img,0,3,0.8)
cv2.imwrite('lenna04_1.png',img1)
