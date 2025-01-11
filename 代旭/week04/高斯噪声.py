import numpy as np
import cv2
from numpy import shape
import random

def GaussianNoise(src,means,sigama,percetage):
    guassnoise = src
    noisecount = int(percetage*src.shape[0]*src.shape[1])
    for i in range(noisecount):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        guassnoise[randX,randY] = guassnoise[randX,randY]+random.gauss(means,sigama)
        if guassnoise[randX,randY]<0:
            guassnoise[randX,randY] = 0
        elif guassnoise[randX,randY]>255:
            guassnoise[randX,randY] = 255
    return guassnoise

def PepperAndSaltNoise(src,percetage):
    pAndS = src
    noisecount = int(percetage*src.shape[0]*src.shape[1])
    for i in range(noisecount):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:
            pAndS[randX,randY]=0
        else:
            pAndS[randX,randY]=255
    return pAndS

img = cv2.imread("lenna.png",0)
img1 = GaussianNoise(img,2,24,0.8)
img = cv2.imread("lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.imread("lenna.png",0)
img3 = PepperAndSaltNoise(img,0.5)

cv2.imshow("source",img2)
cv2.imshow("lenna_Guass",img1)
cv2.imshow("lenna_PS",img3)
cv2.waitKey(0)
