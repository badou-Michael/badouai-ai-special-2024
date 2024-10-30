import numpy as np
import cv2
from numpy import shape 
import random
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])  ##生成高斯噪声的数目
    for i in range(NoiseNum):
		#每次取一个随机点
		#把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        #random.randint生成随机整数
        randX=random.randint(0,src.shape[0]-1) 
        randY=random.randint(0,src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img = cv2.imread('lenna.png',0)  ##以灰度方式读图
img1 = GaussianNoise(img,1,2,0.7)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)
