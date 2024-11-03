import numpy as np
import cv2
from numpy import shape
import random
def GuassNoice(src,means,sigma,percetage):
    GuassImg=src
    GuassNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(GuassNum):
        randomX=random.randint(0,src.shape[0]-1)
        randomY=random.randint(0,src.shape[1]-1)
        GuassImg[randomX,randomY]=GuassImg[randomX,randomY]+random.gauss(means,sigma)
    if GuassImg[randomX,randomY]<0:
        GuassImg[randomX, randomY]=0
    elif GuassImg[randomX,randomY]>255:
        GuassImg[randomX, randomY]=255
    return GuassImg

img=cv2.imread("lenna.png",0)
img1=GuassNoice(img,4,6,0.6)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("GuassNoice",img1)
cv2.imshow("GrayImage",img2)
cv2.waitKey(0)
