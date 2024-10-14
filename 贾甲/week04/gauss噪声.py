#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-09-23
import cv2
import numpy as np
from numpy import shape
import random

def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)

        if NoiseImg[randX,randY]<0:
           NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]>255
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread("lenna.png",0)
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


cv2.imshow('source',img2)
cv2.imshow('lenna,GaussianNoise.png',img1)


cv2.waitKey()

