import cv2
import numpy as np
from numpy import shape
import random

def fun1(image,percetage):
    NoiseImage = image
    NoiseNum = int(percetage*image.shape[0]*image.shape[1])
    for i in range(NoiseNum):
        PointX = random.randint(0,image.shape[0]-1)
        PointY = random.randint(0,image.shape[1]-1)
        if random.random() <=0.5:
            NoiseImage[PointX,PointY] = 0
        else:
            NoiseImage[PointX,PointY] = 255
    return NoiseImage
image = cv2.imread("lenna.png",0)
image1 = fun1(image,0.8)
image = cv2.imread("lenna.png")
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("窗口1",image1)
cv2.imshow("窗口2",image2)
cv2.waitKey()
