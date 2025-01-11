import random #引入random模块
import cv2
import numpy as np

def GaussiNoise(scr,means,sigama,percentage):
    NoiseImage=scr
    NoiseNum=int(scr.shape[0]*scr.shape[1]*percentage)#要加噪的像素数目
    for i in range(NoiseNum):
        # 每次取一个随机点
        # 把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
        # random.randint生成随机整数
        # 高斯噪声图片边缘不处理，故-1
        randX=random.randint(0,scr.shape[0]-1) # 行即高
        randY=random.randint(0,scr.shape[1]-1) # 列即宽
        # 在原有像素值上加上高斯随机数
        NoiseImage[randX,randY]=scr[randX,randY]+random.gauss(means,sigama)
        if NoiseImage[randX,randY]>255:
           NoiseImage[randX, randY]=255
        elif NoiseImage[randX, randY] < 0:
            NoiseImage[randX, randY] = 0
    return NoiseImage

image=cv2.imread('lenna.png',0) # 读取灰度图像
GaussiNoiseImage = GaussiNoise(image,2,4,0.8)
cv2.imshow("show GaussiNoiseImage:",GaussiNoiseImage)
cv2.waitKey(0)
