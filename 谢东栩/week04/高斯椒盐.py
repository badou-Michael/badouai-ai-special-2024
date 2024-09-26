import numpy as np
import cv2
from numpy import shape
import random

print(cv2.__file__)
#高斯噪声
def AddGaussionNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    #这里作随机取点，添加噪声
    for i in range(NoiseNum):
        #随机生成图片的横纵行来进行定位像素点，不处理边缘进行-1
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #在图片的原来像素点与随机数相加
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #边界值界定，不小于0，不大于255
        if  NoiseImg[randX,randY]<0:
            NoiseImg[randX,randY]=0
        elif NoiseImg[randX,randY]>255:
            NoiseImg[randX,randY]=255
    return NoiseImg


#椒盐噪声
def fun1(src,percetage):
    NoiseImgg=src
    NoiseNumm=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNumm):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
      #根据随机数来决定白色还是黑色
        if random.random()<=0.5:
            NoiseImgg[randX,randY]=0
        else:
            NoiseImgg[randX,randY]=255
    return NoiseImgg
  
img=cv2.imread('cs.jpg',0)
img1=AddGaussionNoise(img,2,4,0.8)
img=cv2.imread('cs.jpg')
img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)

img3=cv2.imread('cs.jpg',0)
img4=fun1(img3,1)
img3=cv2.imread('cs.jpg')
img4 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img4)
cv2.imshow('lenna_PepperandSalt',img3)
cv2.waitKey(0)
