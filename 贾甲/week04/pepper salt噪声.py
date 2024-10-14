#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-09-24
import numpy as np
import cv2
from numpy import shape
import random
def fun1(src,percetage):#percentage,意为"百分比;百分率,百分数"
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('lenna.png',0)
img1=fun1(img,0.2)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey()
