#椒盐噪声
#算法思想：随机选取像素值，并将其随机的变成0或255
#中值滤波降噪椒盐噪声，但是对高斯噪声的降噪效果不好
import cv2
import random

def fun(src,percetage):
    NoiseImg = src
    NoiseNum = int(percetage*NoiseImg.shape[0]*NoiseImg.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        if random.random()<= 0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY] = 255
    return NoiseImg
img = cv2.imread('lenna.png',0)
img1 = fun(img,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
cv2.imshow('source',img2)
cv2.imshow("fun",img1)
cv2.waitKey(0)
cv2.destroyWindow()