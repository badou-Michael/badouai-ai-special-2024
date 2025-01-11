import numpy as np 
import cv2 
import random 
from numpy import shape 
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src.copy()
    NoiseNum = int(percetage*shape(src)[0]*shape(src)[1])
    for i in range(NoiseNum):   
        randx = random.randint(0,shape(src)[0]-1)
        randy = random.randint(0,shape(src)[1]-1)
        NoiseImg[randx,randy] = NoiseImg[randx,randy] + random.gauss(means,sigma)
        if NoiseImg[randx, randy]< 0:
            NoiseImg[randx, randy]=0
        elif NoiseImg[randx, randy]>255:
            NoiseImg[randx, randy]=255
    return NoiseImg

img = cv2.imread('lenna.png',0)#读取图像并转换为灰度
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('lenna_GaussianNoise.png',img1)
