# 椒盐噪声
import numpy as np
import cv2
from numpy import shape
import random

def jiaoyan(src,percetage):
    NoiseImg = src
    NoiseSum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseSum):
        sinX = random.randint(0,src.shape[0]-1)
        sinY = random.randint(0,src.shape[1]-1)

        if random.random() <= 0.5:
            NoiseImg[sinX,sinY] = 0
        else: NoiseImg[sinX,sinY] = 255

    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = jiaoyan(img,0.8)
cv2.imwrite('lenna04_2.png',img1)
