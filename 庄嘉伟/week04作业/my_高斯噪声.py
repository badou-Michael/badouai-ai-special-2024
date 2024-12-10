#高斯噪声
# #随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import cv2
import numpy as np
import random

from cv2 import COLOR_BGR2GRAY
from dask.array import shape


#算法思想：每次取一个随机点，用原有点的像素值加上高斯函数的随机数，注意处理边界即可。
#src为传入图片，means,sigma为两个参数，代表均值和标准差，往往根据经验和想要的实验结果去设置
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg = img
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        #每次取一个像素点，用randX表示随机行和用randY随机列
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        #下面这步为在原有灰度图的像素值上加了新的像素值
        NoiseImg[randX,randY] =NoiseImg[randX,randY] + random.gauss(means,sigma)
        #考虑边界
        if NoiseImg[randX,randY] < 0:
            NoiseImg[randX,randY] = 0
        elif NoiseImg[randX,randY] > 255:
            NoiseImg[randX, randY] = 255

    return NoiseImg

img = cv2.imread("lenna.png",0)#1是彩色图（BGR格式），0是灰度图，-1是原图（如果有则包括透明通道）
img1 = GaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
cv2.imshow('source',img2)
cv2.imshow('gaussian',img1)
cv2.waitKey(0)
cv2.destroyWindow()