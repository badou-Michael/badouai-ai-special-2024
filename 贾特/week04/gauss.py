import random

import cv2
import numpy

def GaussNoise(src,means,sigma,percentage):
    NoiseImg = src
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)

        NoiseImg[randX,randY] = NoiseImg[randX,randY] + random.gauss(means,sigma)

        if NoiseImg[randX,randY] <0:
            NoiseImg[randX,randY]=0
        elif NoiseImg[randX,randY] > 255:
            NoiseImg[randX,randY] = 255

    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = GaussNoise(img,9,9,0.9)

cv2.imshow('source',img)
cv2.imshow('GaussNoiseImg',img1)
cv2.waitKey(0)
