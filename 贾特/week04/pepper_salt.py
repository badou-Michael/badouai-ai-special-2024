import cv2
import numpy
import random

def func(src,percentage):
    NoiseImg = src
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])

    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        if random.random() <= 0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY] = 255
    return NoiseImg

img = cv2.imread('lenna.png',0)
img1 = func(img,0.7)

cv2.imshow('jiaoyan',img1)
cv2.waitKey(0)
