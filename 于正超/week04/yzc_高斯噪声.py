"""
高斯噪声-yzc
"""
import random

import numpy as np
import cv2
def GsNoise(src,mean,sigma,percent):
    imgGs = src
    NoiseNum = int(percent * imgGs.shape[0] * imgGs.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0, imgGs.shape[0] - 1)
        randY=random.randint(0,imgGs.shape[1] -1)
        imgGs[randX,randY] = imgGs[randX,randY] + random.gauss(mean,sigma)
        if imgGs[randX,randY] < 0:
            imgGs[randX,randY] = 0
        elif imgGs[randX,randY] > 255:
            imgGs[randX,randY] =255
    return imgGs

img = cv2.imread("..\\lenna.png",0)
imgGs=GsNoise(img,2,4,0.8)
img1 = cv2.imread("..\\lenna.png")
img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
# cv2.imshow("yuantu gray",img)
# cv2.imshow("img gauss", imgGs)
cv2.imshow("yuantu:gauss",np.hstack([img1_gray,img,imgGs]))
cv2.waitKey()
