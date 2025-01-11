import numpy as np
import random
import cv2
from numpy import shape

def GuassNoise(src, sigma, means, percetage):
    noiseimg = src
    noisenum = int(src.shape[0]*src.shape[1]*percetage)
    for i in range(noisenum):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        noiseimg[randX, randY] += random.gauss(sigma, means)
        if noiseimg[randX, randY] < 0:
            noiseimg[randX, randY] = 0
        elif noiseimg[randX, randY] > 255:
            noiseimg[randX, randY] = 255
    return noiseimg

img = cv2.imread("lenna.png", 0)#以单通道的灰度图像读取 ,以原图读取(针对有Alpha通道的图片) -1,以3通道BGR的彩色图片读取 1
img1 = GuassNoise(img, 2, 4, 0.6)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('origin',img2)
cv2.imshow('guassimg', img1)
cv2.waitKey(0)


