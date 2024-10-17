import cv2
import numpy as np
from numpy import shape
import random

def GaussNoise(src,means,sigma,percentage):
    NoiseImg = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randomX = random.randint(0,src.shape[0]-1)
        randomY = random.randint(0,src.shape[1]-1)
        
        NoiseImg[randomX,randomY] = NoiseImg[randomX,randomY] + random.gauss(means,sigma)
        
        if NoiseImg[randomX,randomY] < 0:
            NoiseImg[randomX,randomY] = 0
        elif NoiseImg[randomX,randomY] > 255:
            NoiseImg[randomX,randomY] = 255
    return NoiseImg

img = cv2.imread("lenna.png",0)
img1 = GaussNoise(img,2,4,0.8)
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('sorce',img2)
cv2.imshow('lenna_GaussNoise',img1)
cv2.waitKey()
