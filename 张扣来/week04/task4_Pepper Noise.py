import numpy as np
import cv2
from  numpy import shape
import random

def pepper(src,percetage):
    ImgPepper = src
    ImgNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(ImgNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:
            ImgPepper[randX,randY]=0
        else:
            ImgPepper[randX,randY]=255
    return ImgPepper

img = cv2.imread("../../request/task2/lenna.png",0)
img1 = pepper(img,0.2)

cv2.imshow("PepperNoise",img1)
cv2.waitKey(0)