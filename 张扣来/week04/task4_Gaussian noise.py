import numpy as np
import cv2
from numpy import shape
import random
def GaussianNoise(src,means,sigma,percetage):
    # src:噪声图片  means：高斯均值 sigma：高斯标准差 percetage：像素噪声增加比例
    NoiseImg = src
    # 计算加噪声的像素数量
    NoiseNum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        # 处理边界
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[0]-1)
        # 高斯处理
        NoiseImg[randX,randY]= NoiseImg[randX,randY]+random.gauss(means,sigma)
        # 对超出边间的值进行处理
        if NoiseImg[randX,randY]<0:
           NoiseImg[randX,randY] = 0
        elif NoiseImg[randX,randY]>255:
             NoiseImg[randX,randY] = 255
    return  NoiseImg
# 读灰色图像
img = cv2.imread("../../request/task2/lenna.png",0)
# 拿到img数据，定义函数
img1 = GaussianNoise(img,2,0.4,2)
# img = cv2.imread("../../request/task2/lenna.png")
# img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("source",img2)
cv2.imshow("GaussianNoise",img1)
cv2.waitKey(0)
